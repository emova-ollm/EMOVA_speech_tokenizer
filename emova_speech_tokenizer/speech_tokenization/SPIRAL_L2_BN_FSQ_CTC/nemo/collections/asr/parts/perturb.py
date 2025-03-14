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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter
import copy
import io
import os
import random
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Optional, Union

import librosa
import numpy as np
import pandas
import soundfile as sf
import sox
import webdataset as wd
from omegaconf import DictConfig, OmegaConf
from scipy import signal
from torch.utils.data import IterableDataset

from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.segment import AudioSegment
from nemo.utils import logging

try:
    from nemo.collections.asr.parts import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False


def read_one_audiosegment(manifest, target_sr, rng, tarred_audio=False, audio_dataset=None):

    if tarred_audio:
        if audio_dataset is None:
            raise TypeError("Expected augmentation dataset but got None")
        audio_file, file_id = next(audio_dataset)
        manifest_idx = manifest.mapping[file_id]
        manifest_entry = manifest[manifest_idx]

        offset = 0 if manifest_entry.offset is None else manifest_entry.offset
        duration = 0 if manifest_entry.duration is None else manifest_entry.duration

    else:
        audio_record = rng.sample(manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = 0 if audio_record.offset is None else audio_record.offset
        duration = 0 if audio_record.duration is None else audio_record.duration

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    def __init__(self, sr, resample_type, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, rng=None):
        """
        Performs Speed Augmentation by re-sampling the data to a different sampling rate,
        which does not preserve pitch.

        Note: This is a very slow operation for online augmentation. If space allows,
        it is preferable to pre-compute and save the files to augment the dataset.

        Args:
            sr: Original sampling rate.
            resample_type: Type of resampling operation that will be performed.
                For better speed using `resampy`'s fast resampling method, use `resample_type='kaiser_fast'`.
                For high-quality resampling, set `resample_type='kaiser_best'`.
                To use `scipy.signal.resample`, set `resample_type='fft'` or `resample_type='scipy'`
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or negative
                integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.
            rng: Random seed number.
        """
        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        if resample_type not in ('kaiser_best', 'kaiser_fast', 'fft', 'scipy'):
            raise ValueError("Supported `resample_type` values are ('kaiser_best', 'kaiser_fast', 'fft', 'scipy')")

        self._sr = sr
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        self._res_type = resample_type
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        new_sr = int(self._sr * speed_rate)
        data._samples = librosa.core.resample(data._samples, self._sr, new_sr, res_type=self._res_type)


class TimeStretchPerturbation(Perturbation):
    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, n_fft=512, rng=None):
        """
        Time-stretch an audio series by a fixed rate while preserving pitch, based on [1, 2].

        Note:
        This is a simplified implementation, intended primarily for reference and pedagogical purposes.
        It makes no attempt to handle transients, and is likely to produce audible artifacts.

        Reference
        [1] [Ellis, D. P. W. “A phase vocoder in Matlab.” Columbia University, 2002.]
        (http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/)
        [2] [librosa.effects.time_stretch]
        (https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html)

        Args:
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or negative
                integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to avoid unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.
            n_fft: Number of fft filters to be computed.
            rng: Random seed number.
        """
        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        self._rng = random.Random() if rng is None else rng

        # Pre-compute constants
        self._n_fft = int(n_fft)
        self._hop_length = int(n_fft // 2)

        # Pre-allocate buffers
        self._phi_advance_fast = np.linspace(0, np.pi * self._hop_length, self._hop_length + 1)
        self._scale_buffer_fast = np.empty(self._hop_length + 1, dtype=np.float32)

        self._phi_advance_slow = np.linspace(0, np.pi * self._n_fft, self._n_fft + 1)
        self._scale_buffer_slow = np.empty(self._n_fft + 1, dtype=np.float32)

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        # Increase `n_fft` based on task (speed up or slow down audio)
        # This greatly reduces upper bound of maximum time taken
        # to compute slowed down audio segments.
        if speed_rate >= 1.0:  # Speed up audio
            fft_multiplier = 1
            phi_advance = self._phi_advance_fast
            scale_buffer = self._scale_buffer_fast

        else:  # Slow down audio
            fft_multiplier = 2
            phi_advance = self._phi_advance_slow
            scale_buffer = self._scale_buffer_slow

        n_fft = int(self._n_fft * fft_multiplier)
        hop_length = int(self._hop_length * fft_multiplier)

        # Perform short-term Fourier transform (STFT)
        stft = librosa.core.stft(data._samples, n_fft=n_fft, hop_length=hop_length)

        # Stretch by phase vocoding
        if HAVE_NUMBA:
            stft_stretch = numba_utils.phase_vocoder(stft, speed_rate, phi_advance, scale_buffer)

        else:
            stft_stretch = librosa.core.phase_vocoder(stft, speed_rate, hop_length)

        # Predict the length of y_stretch
        len_stretch = int(round(len(data._samples) / speed_rate))

        # Invert the STFT
        y_stretch = librosa.core.istft(
            stft_stretch, dtype=data._samples.dtype, hop_length=hop_length, length=len_stretch
        )

        data._samples = y_stretch


class TempoPerturbation(Perturbation):
    def __init__(self, factors, rng=None):
        assert len(factors) > 0
        assert min(factors) > 0
        self.factors = factors
        self._max_factor = max(self.factors)
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_factor

    def perturb(self, data):
        speed_rate = self._rng.choice(self.factors)
        if speed_rate == 1.0:
            return

        tfm = sox.Transformer()
        if abs(speed_rate - 1.0) <= 0.1:
            tfm.stretch(speed_rate)
        else:
            tfm.tempo(speed_rate)
        perturbed_data = tfm.build_array(input_array=data._samples, sample_rate_in=data._sample_rate)

        data._samples = perturbed_data


class GainPerturbation(Perturbation):
    """
    Applies random gain to the audio.

    Args:
        min_gain_dbfs (float): Min gain level in dB
        max_gain_dbfs (float): Max gain level in dB
        rng: Random number generator
    """

    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        # logging.debug("gain: %d", gain)
        data._samples = data._samples * (10.0 ** (gain / 20.0))


class ImpulsePerturbation(Perturbation):
    """
    Convolves audio with a Room Impulse Response.

    Args:
        manifest_path (list): Manifest file for RIRs
        audio_tar_filepaths (list): Tar files, if RIR audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        shift_impulse (bool): Shift impulse response to adjust for delay at the beginning
    """

    def __init__(self, manifest_path=None, rng=None, audio_tar_filepaths=None, shuffle_n=128, shift_impulse=False):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._shift_impulse = shift_impulse
        self._data_iterator = None

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse = read_one_audiosegment(
            self._manifest,
            data.sample_rate,
            self._rng,
            tarred_audio=self._tarred_audio,
            audio_dataset=self._data_iterator,
        )
        if not self._shift_impulse:
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            data._samples = signal.fftconvolve(data._samples, impulse_norm, "same")
        else:
            # Find peak and shift peak to left
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            max_ind = np.argmax(np.abs(impulse_norm))

            impulse_resp = impulse_norm[max_ind:]
            delay_after = len(impulse_resp)
            data._samples = signal.fftconvolve(data._samples, impulse_resp, "full")[:-delay_after]


class ShiftPerturbation(Perturbation):
    """
    Perturbs audio by shifting the audio in time by a random amount between min_shift_ms and max_shift_ms.
    The final length of the audio is kept unaltered by padding the audio with zeros.


    Args:
        min_shift_ms (float): Minimum time in milliseconds by which audio will be shifted
        max_shift_ms (float): Maximum time in milliseconds by which audio will be shifted
        rng: Random number generator
    """

    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        # logging.debug("shift: %s", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class NoisePerturbation(Perturbation):
    """
    Perturbation that adds noise to input audio.

    Args:
        manifest_path (str): Manifest file with paths to noise files
        min_snr_db (float): Minimum SNR of audio after noise is added
        max_snr_db (float): Maximum SNR of audio after noise is added
        max_gain_db (float): Maximum gain that can be applied on the noise sample
        audio_tar_filepaths (list) : Tar files, if noise audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        orig_sr (int): Original sampling rate of the noise files
        rng: Random number generator
    """

    def __init__(
        self,
        manifest_path=None,
        min_snr_db=10,
        max_snr_db=50,
        max_gain_db=300.0,
        rng=None,
        audio_tar_filepaths=None,
        shuffle_n=100,
        orig_sr=16000,
    ):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._orig_sr = orig_sr
        self._data_iterator = None

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    @property
    def orig_sr(self):
        return self._orig_sr

    def get_one_noise_sample(self, target_sr):
        return read_one_audiosegment(
            self._manifest, target_sr, self._rng, tarred_audio=self._tarred_audio, audio_dataset=self._data_iterator
        )

    def perturb(self, data):
        noise = read_one_audiosegment(
            self._manifest,
            data.sample_rate,
            self._rng,
            tarred_audio=self._tarred_audio,
            audio_dataset=self._data_iterator,
        )
        self.perturb_with_input_noise(data, noise)

    def perturb_with_input_noise(self, data, noise, data_rms=None):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        if data_rms is None:
            data_rms = data.rms_db
        noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
        # logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record.audio_file)

        # calculate noise segment to use
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        if noise.duration > (start_time + data.duration):
            noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)

        if noise._samples.shape[0] < data._samples.shape[0]:
            noise_idx = self._rng.randint(0, data._samples.shape[0] - noise._samples.shape[0])
            data._samples[noise_idx : noise_idx + noise._samples.shape[0]] += noise._samples

        else:
            data._samples += noise._samples

    def perturb_with_foreground_noise(
        self, data, noise, data_rms=None, max_noise_dur=2, max_additions=1,
    ):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        if not data_rms:
            data_rms = data.rms_db

        noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
        n_additions = self._rng.randint(1, max_additions)

        for i in range(n_additions):
            noise_dur = self._rng.uniform(0.0, max_noise_dur)
            start_time = self._rng.uniform(0.0, noise.duration)
            start_sample = int(round(start_time * noise.sample_rate))
            end_sample = int(round(min(noise.duration, (start_time + noise_dur)) * noise.sample_rate))
            noise_samples = np.copy(noise._samples[start_sample:end_sample])
            # adjust gain for snr purposes and superimpose
            noise_samples *= 10.0 ** (noise_gain_db / 20.0)

            if noise_samples.shape[0] > data._samples.shape[0]:
                noise_samples = noise_samples[0 : data._samples.shape[0]]

            noise_idx = self._rng.randint(0, data._samples.shape[0] - noise_samples.shape[0])
            data._samples[noise_idx : noise_idx + noise_samples.shape[0]] += noise_samples


class RandomNoisePerturbation(Perturbation):
    """
    Perturbation that adds noise to input audio.

    Args:
        manifest_path (str): Manifest file with paths to noise files
        min_snr_db (float): Minimum SNR of audio after noise is added
        max_snr_db (float): Maximum SNR of audio after noise is added
        max_gain_db (float): Maximum gain that can be applied on the noise sample
        audio_tar_filepaths (list) : Tar files, if noise audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        orig_sr (int): Original sampling rate of the noise files
        rng: Random number generator
    """

    def __init__(
        self,
        manifest_path=None,
        min_snr_db=10,
        max_snr_db=50,
        max_gain_db=300.0,
        ratio=1.0,
        rng=None,
        target_sr=16000,
        data_dir='',
        cache_noise=False
    ):

        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db
        self.ratio = ratio

        self.data_dir = data_dir
        self.target_sr = target_sr
        manifest, self._noise_weights = self.read_manifest(manifest_path)
        self.noise_files = manifest['wav_filename'].tolist()
        self._cache = {}
        self.cache_noise = cache_noise

    def read_manifest(self, manifest_fps):
        manifest_files = []
        for fp in manifest_fps:
            manifest_files.append(pandas.read_csv(fp, encoding='utf-8'))
        manifest = pandas.concat(manifest_files)

        orig_noise_num = len(manifest)
        wav_header_size = 44
        # only use noise with duration longer than 1 second
        manifest = manifest[manifest['wav_filesize'] > (1 * 16000 * 2 + wav_header_size)]
        print('filter noise less than 1s: from {} to {} samples'.format(orig_noise_num, len(manifest)))
        wav_data_size = manifest['wav_filesize'].values - wav_header_size
        print('noise duration sum: {}h'.format(wav_data_size.sum() / (16000 * 2) / 60 / 60))
        noise_weights = wav_data_size / wav_data_size.sum()
        return manifest, noise_weights.tolist()

    def perturb(self, data):
        if self._rng.random() < self.ratio:
            noises = self.get_noises(data.num_samples)
            self.perturb_with_input_noise(data, noises)

    def perturb_with_input_noise(self, data, noises):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        data_rms = rms_db(data._samples)

        start_index = 0
        for noise_i in noises:
            noise_gain_db = min(data_rms - rms_db(noise_i) - snr_db, self._max_gain_db)
            # logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record.audio_file)

            # adjust gain for snr purposes and superimpose
            noise_i = gain_db(noise_i, noise_gain_db)

            end_index = start_index + noise_i.shape[0]
            data._samples[start_index:end_index] += noise_i
            start_index = end_index
        assert end_index == data.num_samples

    def get_noises(self, num_samples):
        left_noise_samples = num_samples
        noise_data = []
        while left_noise_samples > 0:
            noise = self.read_one_noise()
            if noise.shape[0] > left_noise_samples:
                start_pos = self._rng.randrange(0, noise.shape[0] - left_noise_samples + 1)
                noise = noise[start_pos:start_pos + left_noise_samples]
            left_noise_samples -= noise.shape[0]
            noise_data.append(noise)
        assert left_noise_samples == 0
        return noise_data

    def read_one_noise(self):
        fp = self._rng.choices(self.noise_files, weights=self._noise_weights)[0]
        fp = os.path.join(self.data_dir, fp)

        cached_noise = self._cache.get(fp, None)
        if cached_noise is None:
            cached_noise = AudioSegment.from_file(fp, target_sr=self.target_sr)._samples
            if self.cache_noise:
                self._cache[fp] = cached_noise
        return cached_noise.copy()


def rms_db(samples):
    mean_square = np.mean(samples ** 2)
    if mean_square == 0:
        return -np.inf
    else:
        return 10 * np.log10(mean_square)


def gain_db(samples, gain):
    return samples * (10.0 ** (gain / 20.0))


class WhiteNoisePerturbation(Perturbation):
    """
    Perturbation that adds white noise to an audio file in the training dataset.

    Args:
        min_level (int): Minimum level in dB at which white noise should be added
        max_level (int): Maximum level in dB at which white noise should be added
        rng: Random number generator
    """

    def __init__(self, min_level=-90, max_level=-46, rng=None):
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self._rng = np.random.RandomState() if rng is None else rng

    def perturb(self, data):
        noise_level_db = self._rng.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = self._rng.randn(data._samples.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        data._samples += noise_signal


class RirAndNoisePerturbation(Perturbation):
    def __init__(
        self,
        rir_manifest_path=None,
        rir_prob=0.5,
        noise_manifest_paths=None,
        min_snr_db=0,
        max_snr_db=50,
        rir_tar_filepaths=None,
        rir_shuffle_n=100,
        noise_tar_filepaths=None,
        apply_noise_rir=False,
        orig_sample_rate=None,
        max_additions=5,
        max_duration=2.0,
        bg_noise_manifest_paths=None,
        bg_min_snr_db=10,
        bg_max_snr_db=50,
        bg_noise_tar_filepaths=None,
        bg_orig_sample_rate=None,
    ):
        """
        RIR augmentation with additive foreground and background noise.
        In this implementation audio data is augmented by first convolving the audio with a Room Impulse Response
        and then adding foreground noise and background noise at various SNRs. RIR, foreground and background noises
        should either be supplied with a manifest file or as tarred audio files (faster).

        Different sets of noise audio files based on the original sampling rate of the noise. This is useful while
        training a mixed sample rate model. For example, when training a mixed model with 8 kHz and 16 kHz audio with a
        target sampling rate of 16 kHz, one would want to augment 8 kHz data with 8 kHz noise rather than 16 kHz noise.

        Args:
            rir_manifest_path: manifest file for RIRs
            rir_tar_filepaths: tar files, if RIR audio files are tarred
            rir_prob: probability of applying a RIR
            noise_manifest_paths: foreground noise manifest path
            min_snr_db: min SNR for foreground noise
            max_snr_db: max SNR for background noise,
            noise_tar_filepaths: tar files, if noise files are tarred
            apply_noise_rir: whether to convolve foreground noise with a a random RIR
            orig_sample_rate: original sampling rate of foreground noise audio
            max_additions: max number of times foreground noise is added to an utterance,
            max_duration: max duration of foreground noise
            bg_noise_manifest_paths: background noise manifest path
            bg_min_snr_db: min SNR for background noise
            bg_max_snr_db: max SNR for background noise
            bg_noise_tar_filepaths: tar files, if noise files are tarred
            bg_orig_sample_rate: original sampling rate of background noise audio

        """
        logging.info("Called Rir aug init")
        self._rir_prob = rir_prob
        self._rng = random.Random()
        self._rir_perturber = ImpulsePerturbation(
            manifest_path=rir_manifest_path,
            audio_tar_filepaths=rir_tar_filepaths,
            shuffle_n=rir_shuffle_n,
            shift_impulse=True,
        )
        self._fg_noise_perturbers = {}
        self._bg_noise_perturbers = {}
        if noise_manifest_paths:
            for i in range(len(noise_manifest_paths)):
                if orig_sample_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = orig_sample_rate[i]
                self._fg_noise_perturbers[orig_sr] = NoisePerturbation(
                    manifest_path=noise_manifest_paths[i],
                    min_snr_db=min_snr_db[i],
                    max_snr_db=max_snr_db[i],
                    audio_tar_filepaths=noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )
        self._max_additions = max_additions
        self._max_duration = max_duration
        if bg_noise_manifest_paths:
            for i in range(len(bg_noise_manifest_paths)):
                if bg_orig_sample_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = bg_orig_sample_rate[i]
                self._bg_noise_perturbers[orig_sr] = NoisePerturbation(
                    manifest_path=bg_noise_manifest_paths[i],
                    min_snr_db=bg_min_snr_db[i],
                    max_snr_db=bg_max_snr_db[i],
                    audio_tar_filepaths=bg_noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )

        self._apply_noise_rir = apply_noise_rir

    def perturb(self, data):
        prob = self._rng.uniform(0.0, 1.0)

        if prob < self._rir_prob:
            self._rir_perturber.perturb(data)

        orig_sr = data.orig_sr
        if orig_sr not in self._fg_noise_perturbers:
            orig_sr = max(self._fg_noise_perturbers.keys())
        fg_perturber = self._fg_noise_perturbers[orig_sr]

        orig_sr = data.orig_sr
        if orig_sr not in self._bg_noise_perturbers:
            orig_sr = max(self._bg_noise_perturbers.keys())
        bg_perturber = self._bg_noise_perturbers[orig_sr]

        data_rms = data.rms_db
        noise = fg_perturber.get_one_noise_sample(data.sample_rate)
        if self._apply_noise_rir:
            self._rir_perturber.perturb(noise)
        fg_perturber.perturb_with_foreground_noise(
            data, noise, data_rms=data_rms, max_noise_dur=self._max_duration, max_additions=self._max_additions
        )
        noise = bg_perturber.get_one_noise_sample(data.sample_rate)
        bg_perturber.perturb_with_input_noise(data, noise, data_rms=data_rms)


class TranscodePerturbation(Perturbation):
    def __init__(self, rng=None):
        """
        Audio codec augmentation. This implementation uses sox to transcode audio with low rate audio codecs,
        so users need to make sure that the installed sox version supports the codecs used here (G711 and amr-nb).

        """
        self._rng = np.random.RandomState() if rng is None else rng
        self._codecs = ["g711", "amr-nb"]

    def perturb(self, data):
        att_factor = 0.8
        max_level = np.max(np.abs(data._samples))
        norm_factor = att_factor / max_level
        norm_samples = norm_factor * data._samples
        orig_f = NamedTemporaryFile(suffix=".wav")
        sf.write(orig_f.name, norm_samples.transpose(), 16000)

        codec_ind = random.randint(0, len(self._codecs) - 1)
        if self._codecs[codec_ind] == "amr-nb":
            transcoded_f = NamedTemporaryFile(suffix="_amr.wav")
            rates = list(range(0, 8))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "g711":
            transcoded_f = NamedTemporaryFile(suffix="_g711.wav")
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0  -r 8000 -c 1 -e a-law {transcoded_f.name}", shell=True
            )

        new_data = AudioSegment.from_file(transcoded_f.name, target_sr=16000)
        data._samples = new_data._samples[0 : data._samples.shape[0]]
        return


perturbation_types = {
    "speed": SpeedPerturbation,
    "time_stretch": TimeStretchPerturbation,
    "tempo": TempoPerturbation,
    "gain": GainPerturbation,
    "impulse": ImpulsePerturbation,
    "shift": ShiftPerturbation,
    "noise": NoisePerturbation,
    "white_noise": WhiteNoisePerturbation,
    "rir_noise_aug": RirAndNoisePerturbation,
    "transcode_aug": TranscodePerturbation,
}


def register_perturbation(name: str, perturbation: Perturbation):
    if name in perturbation_types.keys():
        raise KeyError(
            f"Perturbation with the name {name} exists. " f"Type of perturbation : {perturbation_types[name]}."
        )

    perturbation_types[name] = perturbation


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []
        if perturbations:
            print('audio perturbations:', perturbations)

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                logging.warning("%s perturbation not known. Skipping.", p['aug_type'])
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


def process_augmentations(augmenter) -> Optional[AudioAugmentor]:
    """Process list of online data augmentations.
    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.
    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.
    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].
    # Example in YAML config file
    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.
    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```
    Then in the training script,
    ```python
    import copy
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)
    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )
    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```
    # Registering your own Augmentations
    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.
    ```python
    from nemo.collections.asr.parts import perturb
    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...
    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```
    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    Returns: AudioAugmentor object
    """
    if augmenter is None:
        return None

    if isinstance(augmenter, AudioAugmentor):
        return augmenter

    if not type(augmenter) in {dict, DictConfig}:
        raise ValueError("Cannot parse augmenter. Must be a dict or an AudioAugmentor object ")

    if isinstance(augmenter, DictConfig):
        augmenter = OmegaConf.to_container(augmenter, resolve=True)

    augmenter = copy.deepcopy(augmenter)

    augmentations = []
    for augment_name, augment_kwargs in augmenter.items():
        prob = augment_kwargs.get('prob', None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )

        else:
            _ = augment_kwargs.pop('prob')

            if prob < 0.0 or prob > 1.0:
                raise ValueError("`prob` must be a float value between 0 and 1.")

            try:
                augmentation = perturbation_types[augment_name](**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


class AugmentationDataset(IterableDataset):
    """
        A class that loads tarred audio files and cycles over the files in the dataset.

        Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
        as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
        contain the information for one audio file, including at least the transcript and name of the audio
        file within the tarball.

        Valid formats for the audio_tar_filepaths argument include:
        (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
        (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

        Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
        This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
        Supported opening braces - { <=> (, [, < and the special tag _OP_.
        Supported closing braces - } <=> ), ], > and the special tag _CL_.
        For SLURM based tasks, we suggest the use of the special tags for ease of use.

        See the WebDataset documentation for more information about accepted data and input formats.
    """

    def __init__(self, manifest_path: str, tar_filepaths: Union[str, List[str]], shuffle_n: int = 128):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)

        if isinstance(tar_filepaths, str):
            # Replace '(' and '[' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in tar_filepaths:
                    tar_filepaths = tar_filepaths.replace(bkey, "{")

            # Replace ')' and ']' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in tar_filepaths:
                    tar_filepaths = tar_filepaths.replace(bkey, "}")

        self.audio_dataset = (
            wd.Dataset(tar_filepaths).shuffle(shuffle_n).rename(audio='wav', key='__key__').to_tuple('audio', 'key')
        )
        self.audio_iter = iter(self.audio_dataset)

    def __len__(self):
        return len(self._manifest)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                audio_bytes, audio_filename = next(self.audio_iter)

            except StopIteration:
                self.audio_iter = iter(self.audio_dataset)
                audio_bytes, audio_filename = next(self.audio_iter)
            file_id, _ = os.path.splitext(os.path.basename(audio_filename))

            # Convert audio bytes to IO stream for processing (for SoundFile to read)
            audio_file = io.BytesIO(audio_bytes)
            return audio_file, file_id
