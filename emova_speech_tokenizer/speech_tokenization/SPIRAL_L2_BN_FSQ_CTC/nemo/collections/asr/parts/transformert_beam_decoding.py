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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.asr.modules import rnnt_abstract, TransformerTDecoder
from nemo.collections.asr.parts import rnnt_utils
from nemo.collections.asr.parts.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, HypothesisType, LengthsType, NeuralType


class BeamTransformerTInfer(Typing):
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": NeuralType(elements_type=HypothesisType())}

    def __init__(
        self,
        decoder_model: TransformerTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        beam_size: int,
        beam_temperature: float,
        beam_combine_path: bool,
        beam_max_exp_step: int,
        beam_prune_exp: bool,
        beam_prune_exp_full: bool,
        beam_word_reward_ratio: float,
        search_type: str = 'convtt',
        score_norm: bool = True,
        return_best_hypothesis: bool = True,
        tsd_max_sym_exp_per_step: Optional[int] = 50,
        alsd_max_target_len: Union[int, float] = 1.0,
        nsc_max_timesteps_expansion: int = 1,
        nsc_prefix_alpha: int = 1,
    ):
        """
        Beam Search implementation ported from ESPNet implementation -
        https://github.com/espnet/espnet/blob/master/espnet/nets/beam_search_transducer.py

        Sequence level beam decoding or batched-beam decoding, performed auto-repressively
        depending on the search type chosen.

        Args:
            decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
            joint_model: rnnt_utils.AbstractRNNTJoint implementation.

            beam_size: number of beams for beam search. Must be a positive integer >= 1.
                If beam size is 1, defaults to stateful greedy search.
                This greedy search might result in slightly different results than
                the greedy results obtained by GreedyRNNTInfer due to implementation differences.

                For accurate greedy results, please use GreedyRNNTInfer or GreedyBatchedRNNTInfer.

            search_type: str representing the type of beam search to perform.
                Must be one of ['beam', 'tsd', 'alsd']. 'nsc' is currently not supported.

                Algoritm used:
                `beam` - basic beam search strategy. Larger beams generally result in better decoding,
                    however the time required for the search also grows steadily.

                `tsd` - time synchronous decoding. Please refer to the paper:
                    [Alignment-Length Synchronous Decoding for RNN Transducer](https://ieeexplore.ieee.org/document/9053040)
                    for details on the algorithm implemented.

                    Time synchronous decoding (TSD) execution time grows by the factor T * max_symmetric_expansions.
                    For longer sequences, T is greater, and can therefore take a long time for beams to obtain
                    good results. This also requires greater memory to execute.

                `alsd` - alignment-length synchronous decoding. Please refer to the paper:
                    [Alignment-Length Synchronous Decoding for RNN Transducer](https://ieeexplore.ieee.org/document/9053040)
                    for details on the algorithm implemented.

                    Alignment-length synchronous decoding (ALSD) execution time is faster than TSD, with growth
                    factor of T + U_max, where U_max is the maximum target length expected during execution.

                    Generally, T + U_max < T * max_symmetric_expansions. However, ALSD beams are non-unique,
                    therefore it is required to use larger beam sizes to achieve the same (or close to the same)
                    decoding accuracy as TSD.

                    For a given decoding accuracy, it is possible to attain faster decoding via ALSD than TSD.

            score_norm: bool, whether to normalize the scores of the log probabilities.

            return_best_hypothesis: bool, decides whether to return a single hypothesis (the best out of N),
                or return all N hypothesis (sorted with best score first). The container class changes based
                this flag -
                When set to True (default), returns a single Hypothesis.
                When set to False, returns a NBestHypotheses container, which contains a list of Hypothesis.

            # The following arguments are specific to the chosen `search_type`

            tsd_max_sym_exp_per_step: Used for `search_type=tsd`. The maximum symmetric expansions allowed
                per timestep during beam search. Larger values should be used to attempt decoding of longer
                sequences, but this in turn increases execution time and memory usage.

            alsd_max_target_len: Used for `search_type=alsd`. The maximum expected target sequence length
                during beam search. Larger values allow decoding of longer sequences at the expense of
                execution time and memory.

            # The following two flags are placeholders and unused until `nsc` implementation is stabilized.

            nsc_max_timesteps_expansion: Unused int.

            nsc_prefix_alpha: Unused int.
        """
        self.decoder = decoder_model
        self.joint = joint_model

        self.blank = blank_index
        assert self.blank == self.decoder.blank_idx
        self.vocab_size = decoder_model.vocab_size
        self.search_type = search_type
        self.return_best_hypothesis = return_best_hypothesis

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size
        self.score_norm = score_norm

        self.beam_stepwise_ln_alpha = 0.
        self.beam_word_reward_ratio = beam_word_reward_ratio
        if self.beam_stepwise_ln_alpha > 0:
            assert self.score_norm
            assert self.beam_word_reward_ratio == 0
        if self.beam_word_reward_ratio > 0:
            assert self.score_norm
            assert self.beam_stepwise_ln_alpha == 0
        self.beam_combine_path = beam_combine_path
        self.beam_max_exp_step = beam_max_exp_step
        self.beam_temperature = beam_temperature
        self.beam_prune_exp = beam_prune_exp
        self.beam_prune_exp_full = beam_prune_exp_full

        assert self.beam_size > 1 and self.search_type == 'convtt'

        if self.beam_size == 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "convtt":
            self.search_algorithm = self.convtt_beam_search
        elif search_type == "nsc":
            raise NotImplementedError("`nsc` (Constrained Beam Search) has not been implemented.")
            # self.search_algorithm = self.nsc_beam_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, tsd, alsd, nsc)"
            )

        if tsd_max_sym_exp_per_step is None:
            tsd_max_sym_exp_per_step = -1

        if search_type in ['tsd', 'alsd', 'nsc'] and not self.decoder.blank_as_pad:
            raise ValueError(
                f"Search type was chosen as '{search_type}', however the decoder module provided "
                f"does not support the `blank` token as a pad value. {search_type} requires "
                f"the blank token as pad value support in order to perform batched beam search."
                f"Please chose one of the other beam search methods, or re-train your model "
                f"with this support."
            )

        self.tsd_max_symmetric_expansion_per_step = tsd_max_sym_exp_per_step
        self.alsd_max_target_length = alsd_max_target_len
        self.nsc_max_timesteps_expansion = nsc_max_timesteps_expansion
        self.nsc_prefix_alpha = nsc_prefix_alpha

    @typecheck()
    def __call__(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor
    ) -> Union[Hypothesis, NBestHypotheses]:
        """Perform general beam search.

        Args:
            encoder_output: Encoded speech features (B, T_max, D_enc)
            encoded_lengths: Lengths of the encoder outputs

        Returns:
            Either a list containing a single Hypothesis (when `return_best_hypothesis=True`,
            otherwise a list containing a single NBestHypotheses, which itself contains a list of
            Hypothesis. This list is sorted such that the best hypothesis is the first element.
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training
        self.decoder.eval()
        self.joint.eval()

        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            hypotheses = []
            with tqdm(
                range(encoder_output.size(0)),
                desc='Beam search progress:',
                total=encoder_output.size(0),
                unit='sample',
            ) as idx_gen:
                # Decode every sample in the batch independently.
                for batch_idx in idx_gen:
                    inseq = encoder_output[batch_idx : batch_idx + 1, :, :]  # [1, T, D]
                    logitlen = encoded_lengths[batch_idx]

                    # Execute the specific search strategy
                    nbest_hyps = self.search_algorithm(inseq, logitlen)  # sorted list of hypothesis

                    # Pack the result
                    if self.return_best_hypothesis:
                        best_hypothesis = nbest_hyps[0]  # type: Hypothesis
                    else:
                        best_hypothesis = NBestHypotheses(nbest_hyps)  # type: NBestHypotheses
                    hypotheses.append(best_hypothesis)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (hypotheses,)

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses
        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.y_sequence), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def greedy_search(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transducer.
        Generic case when beam size = 1. Results might differ slightly due to implementation details
        as compared to `GreedyRNNTInfer` and `GreedyBatchRNNTInfer`.

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            hyp: 1-best decoding results
        """
        # Initialize zero state vectors
        dec_state = self.decoder.initialize_state(h)

        # Construct initial hypothesis
        hyp = Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state)
        cache = {}

        # Initialize state and first token
        y, state, _ = self.decoder.score_hypothesis(hyp, cache)

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]

            not_blank = True
            symbols_added = 0

            while not_blank:
                ytu = torch.log_softmax(self.joint.joint(hi, y), dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # max() requires float
                if ytu.dtype != torch.float32:
                    ytu = ytu.float()

                logp, pred = torch.max(ytu, dim=-1)  # [1, 1]
                pred = pred.item()

                if pred == self.blank:
                    not_blank = False
                else:
                    # Update state and current sequence
                    hyp.y_sequence.append(int(pred))
                    hyp.score += float(logp)
                    hyp.dec_state = state

                    # Compute next state and token
                    y, state, _ = self.decoder.score_hypothesis(hyp, cache)
                symbols_added += 1

        return [hyp]

    def default_beam_search(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # Initialize states
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor([self.blank], device=h.device, dtype=torch.long)

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # Initialize zero vector states
        dec_state = self.decoder.initialize_state(h)

        # Initialize first hypothesis for the beam (blank)
        kept_hyps = [Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state)]
        cache = {}

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]  # [1, 1, D]
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                # update decoder state and get next score
                y, state, lm_tokens = self.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

                # get next token
                ytu = torch.log_softmax(self.joint.joint(hi, y), dim=-1)  # [1, 1, 1, V + 1]
                ytu = ytu[0, 0, 0, :]  # [V + 1]

                # remove blank token before top k
                top_k = ytu[ids].topk(beam_k, dim=-1)

                # Two possible steps - blank token or non-blank token predicted
                ytu = (
                    torch.cat((top_k[0], ytu[self.blank].unsqueeze(0))),
                    torch.cat((top_k[1] + index_incr, blank_tensor)),
                )

                # for each possible step
                for logp, k in zip(*ytu):
                    # construct hypothesis for step
                    new_hyp = Hypothesis(
                        score=(max_hyp.score + float(logp)),
                        y_sequence=max_hyp.y_sequence[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )

                    # if current token is blank, dont update sequence, just store the current hypothesis
                    if k == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        # if non-blank token was predicted, update state and sequence and then search more hypothesis
                        new_hyp.dec_state = state
                        new_hyp.y_sequence.append(int(k))

                        hyps.append(new_hyp)

                # keep those hypothesis that have scores greater than next search generation
                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)

                # If enough hypothesis have scores greater than next search generation,
                # stop beam search.
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # prepare the batched beam states
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.initialize_state(
            torch.zeros(beam, device=h.device, dtype=h.dtype)
        )  # [L, B, H], [L, B, H] (for LSTMs)

        # Initialize first hypothesis for the beam (blank)
        B = [Hypothesis(y_sequence=[self.blank], score=0.0, dec_state=self.decoder.batch_select_state(beam_state, 0))]
        cache = {}

        for i in range(int(encoded_lengths)):
            hi = h[:, i : i + 1, :]

            # Update caches
            A = []
            C = B

            h_enc = hi

            # For a limited number of symmetric expansions per timestep "i"
            for v in range(self.tsd_max_symmetric_expansion_per_step):
                D = []

                # Decode a batch of beam states and scores
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(C, cache, beam_state)

                # Extract the log probabilities and the predicted tokens
                beam_logp = torch.log_softmax(self.joint.joint(h_enc, beam_y), dim=-1)  # [B, 1, 1, V + 1]
                beam_logp = beam_logp[:, 0, 0, :]  # [B, V + 1]
                beam_topk = beam_logp[:, ids].topk(beam, dim=-1)

                seq_A = [h.y_sequence for h in A]

                for j, hyp in enumerate(C):
                    # create a new hypothesis in A
                    if hyp.y_sequence not in seq_A:
                        # If the sequence is not in seq_A, add it as the blank token
                        # In this step, we dont add a token but simply update score
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[j, self.blank])),
                                y_sequence=hyp.y_sequence[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        # merge the existing blank hypothesis score with current score.
                        dict_pos = seq_A.index(hyp.y_sequence)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[j, self.blank]))
                        )

                if v < self.tsd_max_symmetric_expansion_per_step:
                    for j, hyp in enumerate(C):
                        # for each current hypothesis j
                        # extract the top token score and top token id for the jth hypothesis
                        for logp, k in zip(beam_topk[0][j], beam_topk[1][j] + index_incr):
                            # create new hypothesis and store in D
                            # Note: This loop does *not* include the blank token!
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                y_sequence=(hyp.y_sequence + [int(k)]),
                                dec_state=self.decoder.batch_select_state(beam_state, j),
                                lm_state=hyp.lm_state,
                            )

                            D.append(new_hyp)

                # Prune beam
                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            # Prune beam
            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        if self.blank == 0:
            index_incr = 1
        else:
            index_incr = 0

        # prepare the batched beam states
        beam = min(self.beam_size, self.vocab_size)

        h = h[0]  # [T, D]
        h_length = int(encoded_lengths)
        beam_state = self.decoder.initialize_state(
            torch.zeros(beam, device=h.device, dtype=h.dtype)
        )  # [L, B, H], [L, B, H] for LSTMS

        # compute u_max as either a specific static limit,
        # or a multiple of current `h_length` dynamically.
        if type(self.alsd_max_target_length) == float:
            u_max = int(self.alsd_max_target_length * h_length)
        else:
            u_max = int(self.alsd_max_target_length)

        # Initialize first hypothesis for the beam (blank)
        B = [Hypothesis(y_sequence=[self.blank], score=0.0, dec_state=self.decoder.batch_select_state(beam_state, 0))]

        final = []
        cache = {}

        # ALSD runs for T + U_max steps
        for i in range(h_length + u_max):
            # Update caches
            A = []
            B_ = []
            h_states = []

            # preserve the list of batch indices which are added into the list
            # and those which are removed from the list
            # This is necessary to perform state updates in the correct batch indices later
            batch_ids = list(range(len(B)))  # initialize as a list of all batch ids
            batch_removal_ids = []  # update with sample ids which are removed

            for bid, hyp in enumerate(B):
                u = len(hyp.y_sequence) - 1
                t = i - u + 1

                if t > (h_length - 1):
                    batch_removal_ids.append(bid)
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

            if B_:
                # Compute the subset of batch ids which were *not* removed from the list above
                sub_batch_ids = None
                if len(B_) != beam:
                    sub_batch_ids = batch_ids
                    for id in batch_removal_ids:
                        # sub_batch_ids contains list of ids *that were not removed*
                        sub_batch_ids.remove(id)

                    # extract the states of the sub batch only.
                    beam_state_ = [beam_state[state_id][:, sub_batch_ids, :] for state_id in range(len(beam_state))]
                else:
                    # If entire batch was used (none were removed), simply take all the states
                    beam_state_ = beam_state

                # Decode a batch/sub-batch of beam states and scores
                beam_y, beam_state_, beam_lm_tokens = self.decoder.batch_score_hypothesis(B_, cache, beam_state_)

                # If only a subset of batch ids were updated (some were removed)
                if sub_batch_ids is not None:
                    # For each state in the RNN (2 for LSTM)
                    for state_id in range(len(beam_state)):
                        # Update the current batch states with the sub-batch states (in the correct indices)
                        # These indices are specified by sub_batch_ids, the ids of samples which were updated.
                        beam_state[state_id][:, sub_batch_ids, :] = beam_state_[state_id][...]
                else:
                    # If entire batch was updated, simply update all the states
                    beam_state = beam_state_

                # h_states = list of [t, h[t]]
                # so h[1] here is a h[t] of shape [D]
                # Simply stack all of the h[t] within the sub_batch/batch (T <= beam)
                h_enc = torch.stack([h[1] for h in h_states])  # [T=beam, D]
                h_enc = h_enc.unsqueeze(1)  # [B=beam, T=1, D]; batch over the beams

                # Extract the log probabilities and the predicted tokens
                beam_logp = torch.log_softmax(self.joint.joint(h_enc, beam_y), dim=-1)  # [B=beam, 1, 1, V + 1]
                beam_logp = beam_logp[:, 0, 0, :]  # [B=beam, V + 1]
                beam_topk = beam_logp[:, ids].topk(beam, dim=-1)

                for j, hyp in enumerate(B_):
                    # For all updated samples in the batch, add it as the blank token
                    # In this step, we dont add a token but simply update score
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[j, self.blank])),
                        y_sequence=hyp.y_sequence[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    # Add blank prediction to A
                    A.append(new_hyp)

                    # If the prediction "timestep" t has reached the length of the input sequence
                    # we can add it to the "finished" hypothesis list.
                    if h_states[j][0] == (h_length - 1):
                        final.append(new_hyp)

                    # Here, we carefully select the indices of the states that we want to preserve
                    # for the next token (non-blank) update.
                    if sub_batch_ids is not None:
                        h_states_idx = sub_batch_ids[j]
                    else:
                        h_states_idx = j

                    # for each current hypothesis j
                    # extract the top token score and top token id for the jth hypothesis
                    for logp, k in zip(beam_topk[0][j], beam_topk[1][j] + index_incr):
                        # create new hypothesis and store in A
                        # Note: This loop does *not* include the blank token!
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            y_sequence=(hyp.y_sequence[:] + [int(k)]),
                            dec_state=self.decoder.batch_select_state(beam_state, h_states_idx),
                            lm_state=hyp.lm_state,
                        )

                        A.append(new_hyp)

                # Prune and recombine same hypothesis
                # This may cause next beam to be smaller than max beam size
                # Therefore larger beam sizes may be required for better decoding.
                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = self.recombine_hypotheses(B)

            # If B_ is empty list, then we may be able to early exit
            elif len(batch_ids) == len(batch_removal_ids):
                break

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def recombine_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with equivalent output sequence.

        Args:
            hypotheses (list): list of hypotheses

        Returns:
           final (list): list of recombined hypotheses
        """
        final = []

        for hyp in hypotheses:
            seq_final = [f.y_sequence for f in final if f.y_sequence]

            if hyp.y_sequence in seq_final:
                seq_pos = seq_final.index(hyp.y_sequence)

                final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
            else:
                final.append(hyp)

        return hypotheses

    def convtt_beam_search(self, encoder_outputs: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        assert self.decoder.prepend_sos_label
        start_token_id = self.decoder.sos_idx

        init_pred_net_state = self.decoder.initialize_state(1, encoder_outputs.dtype, encoder_outputs.device)

        kept_hyps = [Hyp(score=1.0, labels=tuple(),
                         pred_net_state=init_pred_net_state,
                         last_label=start_token_id)]
        blank_label_id = self.decoder.blank_idx
        assert start_token_id != blank_label_id

        pred_net_cache = {}
        pred_net_cache_steps = 0
        pred_net_cache_missing_steps = 0

        for t in range(int(encoded_lengths)):
            blank_hyps = {}
            exp_i = 0
            exp_hyps = kept_hyps

            while exp_i < self.beam_max_exp_step and len(exp_hyps) > 0:
                new_hyps = []
                # expand hyps
                assert len(exp_hyps) <= self.beam_size
                for hyp_i in exp_hyps:
                    if hyp_i.last_label == blank_label_id:
                        assert hyp_i.pred_net_output is not None
                        pred_net_output = hyp_i.pred_net_output
                        pred_net_state = hyp_i.pred_net_state
                    else:
                        assert hyp_i.last_label == start_token_id or hyp_i.last_label == hyp_i.labels[-1]
                        if hyp_i.labels in pred_net_cache:
                            pred_net_cache_steps += 1
                            pred_net_output, pred_net_state = pred_net_cache[hyp_i.labels]
                        else:
                            pred_net_cache_missing_steps += 1
                            last_label_t = torch.tensor([[hyp_i.last_label]], dtype=torch.long, device=encoder_outputs.device)
                            pred_net_output, pred_net_state = self.decoder.predict(last_label_t,
                                                                                   hyp_i.pred_net_state,
                                                                                   add_sos=False)
                            pred_net_cache[hyp_i.labels] = (pred_net_output, pred_net_state)
                    # [B, T, U, V + 1]
                    logits = self.joint.joint(encoder_outputs[:, t:t+1, :], pred_net_output, apply_softmax=False)
                    if self.beam_temperature != 1.0:
                        logits = logits / self.beam_temperature
                    new_hyp_logp = torch.log_softmax(logits, dim=-1)
                    new_hyp_logp = new_hyp_logp.cpu().numpy()
                    # new_hyp_logp = new_hyp_logp.astype(np.float64)
                    new_hyp_logp += hyp_i.score
                    top_new_hyp_idx = np.argsort(new_hyp_logp)
                    assert top_new_hyp_idx.shape[:3] == (1, 1, 1)
                    top_new_hyp_idx = top_new_hyp_idx[0, 0, 0, -(self.beam_size + 1):]
                    for label_i in top_new_hyp_idx:
                        if label_i == blank_label_id:
                            continue
                        new_hyp = Hyp(score=new_hyp_logp[0, 0, 0, label_i],
                                      labels=hyp_i.labels + (label_i,),
                                      pred_net_state=pred_net_state,
                                      pred_net_output=pred_net_output,
                                      last_label=label_i)
                        new_hyps.append(new_hyp)
                    new_blank_hyp = Hyp(score=new_hyp_logp[0, 0, 0, blank_label_id],
                                        labels=hyp_i.labels,
                                        pred_net_state=pred_net_state,
                                        pred_net_output=pred_net_output,
                                        last_label=blank_label_id)
                    if self.beam_prune_exp:
                        new_hyps.append(new_blank_hyp)
                    if new_blank_hyp.labels in blank_hyps:
                        if self.beam_combine_path:
                            existing_blank_hyp = blank_hyps[new_blank_hyp.labels]
                            existing_blank_hyp.score = np.logaddexp(existing_blank_hyp.score, new_blank_hyp.score)
                        else:
                            if new_blank_hyp.score > blank_hyps[new_blank_hyp.labels].score:
                                blank_hyps[new_blank_hyp.labels] = new_blank_hyp
                    else:
                        blank_hyps[new_blank_hyp.labels] = new_blank_hyp

                if self.beam_prune_exp_full and exp_i > 0:
                    exp_beam_size = len(exp_hyps)
                else:
                    exp_beam_size = self.beam_size

                # one expansion step for all beam finished, select top K hyp for further search
                exp_hyps = get_top_k_hyps(new_hyps, exp_beam_size, ln_alpha=self.beam_stepwise_ln_alpha,
                                          word_reward_ratio=self.beam_word_reward_ratio)
                if self.beam_prune_exp:
                    # orig_exp_hyp_num = len(exp_hyps)
                    exp_hyps = [exp_hyp_i for exp_hyp_i in exp_hyps if exp_hyp_i.last_label != blank_label_id]
                    # print('pruned exp_hyps from {} to {}'.format(orig_exp_hyp_num, len(exp_hyps)))
                exp_i += 1
            # expand finished
            blank_hyps = list(blank_hyps.values())
            kept_hyps = get_top_k_hyps(blank_hyps, self.beam_size, always_sort=True,
                                       ln_alpha=self.beam_stepwise_ln_alpha,
                                       word_reward_ratio=self.beam_word_reward_ratio)

        if self.score_norm:
            if self.beam_stepwise_ln_alpha > 0:
                scores_len_norm = [hyp_i.length_norm_score(self.beam_stepwise_ln_alpha) for hyp_i in kept_hyps]
            elif self.beam_word_reward_ratio > 0:
                scores_len_norm = [hyp_i.word_reward_score(self.beam_word_reward_ratio) for hyp_i in kept_hyps]
            else:
                len_normed_best_hyp, scores_len_norm = get_length_normed_best(kept_hyps)
        else:
            scores_len_norm = [0] * len(kept_hyps)

        ret_hyps = []
        for i in range(len(kept_hyps)):
            ret_hyps.append(Hypothesis(score=scores_len_norm[i] if self.score_norm else kept_hyps[i].score,
                                       y_sequence=[int(label_i) for label_i in kept_hyps[i].labels]))
        return sorted(ret_hyps, key=lambda x: x.score, reverse=True)


class Hyp:
  __slots__ = ('score', 'labels', 'pred_net_state', 'pred_net_output', 'last_label')

  def __init__(self, score, labels=None, pred_net_state=None, pred_net_output=None, last_label=None):
    self.score = score
    self.labels = labels
    self.pred_net_state = pred_net_state
    self.pred_net_output = pred_net_output
    self.last_label = last_label

  def length_norm_score(self, alpha):
    return length_norm(self.score, len(self.labels), alpha)

  def word_reward_score(self, reward_ratio):
    label_len = len(self.labels)
    # if not self.blank:
    #   label_len -= 1
    return self.score + label_len * reward_ratio


def length_norm(log_prob, label_len, alpha):
  len_norm = (label_len + 5.) / 6.
  if alpha != 1:
    len_norm = len_norm ** alpha
  return log_prob / len_norm


def get_top_k_hyps(hyps, k, always_sort=False, ln_alpha=0., word_reward_ratio=0.):
  if len(hyps) <= k and not always_sort:
    return hyps
  else:
    if ln_alpha > 0:
      hyp_scores = np.array([hyp_i.length_norm_score(ln_alpha) for hyp_i in hyps])
    elif word_reward_ratio > 0:
      hyp_scores = np.array([hyp_i.word_reward_score(word_reward_ratio) for hyp_i in hyps])
    else:
      hyp_scores = np.array([hyp_i.score for hyp_i in hyps])
    top_k_idx = np.argsort(hyp_scores)[-k:]
    return [hyps[i] for i in top_k_idx]


def get_length_normed_best(hyps):
  best_hyp = None
  best_hyp_score = None
  score_len_normed = []
  for hyp_i in hyps:
    length_normed_score = hyp_i.score / (len(hyp_i.labels) + 1e-16)
    score_len_normed.append(length_normed_score)
    if best_hyp_score is None or length_normed_score > best_hyp_score:
      best_hyp_score = length_normed_score
      best_hyp = hyp_i
  assert best_hyp is not None
  return best_hyp, score_len_normed
