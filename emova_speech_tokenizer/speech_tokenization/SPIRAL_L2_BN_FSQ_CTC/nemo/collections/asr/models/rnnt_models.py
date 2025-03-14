# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER, RNNTDecoding
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTDecoder
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType, \
    VoidType
from nemo.utils import logging

try:
    import warprnnt_pytorch as warprnnt

    WARP_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WARP_RNNT_AVAILABLE = False


class EncDecRNNTModel(ASRModel):
    """Base class for encoder decoder RNNT-based models."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Required loss function
        if not WARP_RNNT_AVAILABLE:
            raise ImportError(
                "Could not import `warprnnt_pytorch`.\n"
                "Please visit https://github.com/HawkAaron/warp-transducer "
                "and follow the steps in the readme to build and install the "
                "pytorch bindings for RNNT Loss, or use the provided docker "
                "container that supports RNN-T loss."
            )

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        # Update config values required by components dynamically
        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(cfg.labels)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(cfg.labels)
            cfg.joint.vocabulary = cfg.labels
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden
            cfg.joint.blank_pos = cfg.decoder.blank_pos
            
        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize components
        self.preprocessor = EncDecRNNTModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecRNNTModel.from_config_dict(self.cfg.encoder)


        self.decoder: AbstractRNNTDecoder = EncDecRNNTModel.from_config_dict(self.cfg.decoder)
        self.joint = EncDecRNNTModel.from_config_dict(self.cfg.joint)
        self.loss = RNNTLoss(blank_idx=self.decoder.blank_idx)

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecRNNTModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup decoding objects
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
        )
        # Setup WER calculation
        self.wer = RNNTWER(
            decoding=self.decoding, batch_dim_index=0, use_cer=False, log_prediction=True, dist_sync_on_step=True
        )

        # Whether to compute loss during evaluation
        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # setting up the variational noise for the decoder
        if hasattr(self.cfg, 'variational_noise'):
            self._optim_variational_noise_std = self.cfg['variational_noise'].get('std', None)
            self._optim_variational_noise_start = self.cfg['variational_noise'].get('start_step', 0)
        else:
            self._optim_variational_noise_std = 0
            self._optim_variational_noise_start = 0

    @torch.no_grad()
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            paths2audio_files: (a list) of paths to audio files. \
        Recommended length per file is between 5 and 25 seconds. \
        But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
        Bigger will result in better throughput performance but would use more memory.

        Returns:

            A list of transcriptions in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}
        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        try:
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in temporary_datalayer:
                    encoded, encoded_len = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    hypotheses += self.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)
        return hypotheses

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if self.joint.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.joint.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')

            joint_config = self.joint.to_config_dict()
            new_joint_config = copy.deepcopy(joint_config)
            new_joint_config['vocabulary'] = new_vocabulary
            new_joint_config['num_classes'] = len(new_vocabulary)
            del self.joint
            self.joint = EncDecRNNTModel.from_config_dict(new_joint_config)

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config.vocab_size = len(new_vocabulary)
            del self.decoder
            self.decoder = EncDecRNNTModel.from_config_dict(new_decoder_config)

            del self.loss
            self.loss = RNNTLoss(blank_idx=self.decoder.blank_idx)

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            self.decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
            )

            self.wer = RNNTWER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer:
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            # Update config
            with open_dict(self.cfg.joint):
                self.cfg.joint = new_joint_config

            with open_dict(self.cfg.decoder):
                self.cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_char_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "extra": NeuralType(elements_type=VoidType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len, extra = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len, extra

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len, extra = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len, extra = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step

        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        tensorboard = self.logger.experiment

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            tensorboard.add_scalar('train_loss', loss_value, self.global_step)
            tensorboard.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self.global_step)

            # if (sample_id + 1) % log_every_n_steps == 0:
                # self.wer.update(encoded, encoded_len, transcript, transcript_len)
                # _, scores, words = self.wer.compute()
                # tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            tensorboard.add_scalar('train_loss', loss_value, self.global_step)
            tensorboard.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self.global_step)

            if compute_wer:
                tensorboard.add_scalar('training_batch_wer', wer, self.global_step)

        if extra[0] is not None:
            adaptive_ffn_loss = extra[0]
            tensorboard.add_scalar('train_adaffn_loss', adaptive_ffn_loss, self.global_step)
            loss_value = loss_value + adaptive_ffn_loss

        return {'loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0, decode_results=None):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len, extra = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len, extra = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        log_prediction = batch_idx < 3

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(encoded, encoded_len, transcript, transcript_len, decode_results=decode_results, log_prediction=log_prediction)
            wer, wer_num, wer_denom = self.wer.compute()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
                decode_results=decode_results,
                log_prediction=log_prediction
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        if extra[0] is not None:
            tensorboard_logs['val_adaffn_loss'] = extra[0]

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        decode_results = {}
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, decode_results=decode_results)
        test_logs = {
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            # 'test_wer': logs['val_wer'],
            'test_references': decode_results['references'],
            'test_hypotheses': decode_results['hypotheses'],
        }
        if 'val_loss' in logs:
            test_logs['test_loss'] = logs['val_loss']
        if 'val_adaffn_loss' in logs:
            test_logs['test_adaffn_loss'] = logs['val_adaffn_loss']
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        if 'val_adaffn_loss' in outputs[0]:
            adaffn_loss = torch.stack([x['val_adaffn_loss'] for x in outputs]).mean()
            tensorboard_logs['val_adaffn_loss'] = adaffn_loss
        return {**val_loss_log, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}
        references = itertools.chain.from_iterable([x['test_references'] for x in outputs])
        hypotheses = itertools.chain.from_iterable([x['test_hypotheses'] for x in outputs])
        if 'test_adaffn_loss' in outputs[0]:
            adaffn_loss = torch.stack([x['test_adaffn_loss'] for x in outputs]).mean()
            tensorboard_logs['test_adaffn_loss'] = adaffn_loss
        return {**test_loss_log, 'log': tensorboard_logs, 'decode_results': (references, hypotheses)}

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'trim_silence': True,
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def on_after_backward(self):
        super().on_after_backward()
        if self._optim_variational_noise_std > 0 and self.global_step >= self._optim_variational_noise_start:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self._optim_variational_noise_std,
                        size=param.size(),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    param.grad.data.add_(noise)
