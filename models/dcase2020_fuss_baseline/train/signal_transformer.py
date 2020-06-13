# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines SignalTransformer class for converting among signal representations.

stft:
 - (batch,          time) waveform => (batch,          frame, bin) spectrogram
 - (batch, channel, time) waveform => (batch, channel, frame, bin) spectrogram

inverse_stft:
 - (batch,          frame, bin) spectrogram => (batch,          time) waveform
 - (batch, channel, frame, bin) spectrogram => (batch, channel, time) waveform
"""

import numpy as np
import tensorflow.compat.v1 as tf
import math
import librosa as lib
tf.enable_eager_execution()

from . import signal_util


def sqrt_hann_tensor(window_length, dtype):
  """Square-root Hann window as a Tensor. Must match sqrt_hann_array."""
  return tf.sqrt(tf.signal.hann_window(window_length, dtype=dtype,
                                       periodic=True))


class SignalTransformer(object):
  """SignalTransformer converts among signal representations.

  From a complex spectrogram, SignalTransformer can compute other
  representations (e.g., various kinds of spectrograms).
  """

  def __init__(self,
               sample_rate,
               window_time_seconds=0.025,
               hop_time_seconds=0.01,
               magnitude_offset=1e-8,
               zeropad_beginning=False,
               num_basis=-1):
    assert magnitude_offset >= 0, 'magnitude_offset must be nonnegative.'

    self.sample_rate = sample_rate
    self.magnitude_offset = magnitude_offset
    self.zeropad_beginning = zeropad_beginning

    # Compute derivative parameters.
    self.samples_per_window = int(round(sample_rate * window_time_seconds))
    self.hop_time_samples = int(round(self.sample_rate * hop_time_seconds))

    if num_basis <= 0:
      self.fft_len = signal_util.enclosing_power_of_two(self.samples_per_window)
    else:
      assert num_basis >= self.samples_per_window
      self.fft_len = num_basis
    self.fft_bins = int(self.fft_len / 2 + 1)

  def pad_beginning(self, waveform):
    pad_len = int(self.samples_per_window - self.hop_time_samples)
    pad_spec = [(0, 0)] * (len(waveform.shape) - 1) + [(pad_len, 0)]
    return tf.pad(waveform, pad_spec)

  def clip_beginning(self, waveform):
    clip = int(self.samples_per_window - self.hop_time_samples)
    return waveform[..., clip:]

  def forward(self, waveform):
    return self._stft(waveform)

  def inverse(self, spectrogram):
    return self._inverse_stft(spectrogram)

  def forward_mfcc(self,spectrogram,is_graph):
      return self._mfcc_feature(spectrogram,is_graph)

  def _stft(self, waveform):
    """Compute forward STFT with tf.signal, with optional padding on ends."""
    if self.zeropad_beginning:
      waveform = self.pad_beginning(waveform)
    return tf.signal.stft(
        waveform,
        np.int32(self.samples_per_window),
        np.int32(self.hop_time_samples),
        self.fft_len,
        window_fn=sqrt_hann_tensor,
        pad_end=True,
        name='complex_spectrogram')

  def _inverse_stft(self, complex_spectrogram):
    """Compute inverse STFT with tf.signal, with optional padding on ends."""
    waveform = tf.signal.inverse_stft(
        complex_spectrogram,
        self.samples_per_window,
        self.hop_time_samples,
        self.fft_len,
        window_fn=tf.signal.inverse_stft_window_fn(
            self.hop_time_samples, forward_window_fn=sqrt_hann_tensor))
    if self.zeropad_beginning:
      waveform = self.clip_beginning(waveform)
    return waveform

  def _mfcc_feature(self,spectrogram,is_graph):
      if not is_graph:
          mel = tf.zeros(shape=[1,1,1250,128])
          return mel
      power = spectrogram ** 2
      # todo 注释掉了sess
      sess = tf.Session()
      power = sess.run(power)
      for i in range(int(spectrogram.shape[0])):
          temp = lib.feature.melspectrogram(S=power[i, 0].T)
          temp = temp.T
          if __name__ == '__main__':
              temp = temp.T
          temp = tf.expand_dims(temp, axis=0)
          if i == 0:
              mel = temp
          else:
              mel = tf.concat([mel, temp], axis=0)
      mel = tf.expand_dims(mel, axis=1)
      #mfcc = tf.concat([spectrogram, mel], axis=-1)
      return mel

  # def _compute_mfcc_tf(self,spectrogram,
  #                     frame_length=257, sample_rate=16000,
  #                     nfilt=390,
  #                     num_ceps=128):
  #     spec_shape = spectrogram.shape
  #     frame_nums = spec_shape[2]
  #     lifts = []
  #     for n in range(1, num_ceps + 1):
  #         lift = 1 + math.floor(num_ceps) * np.sin(np.pi * n / 12)
  #         lifts.append(lift)
  #     # print(lifts)
  #     for i in range(int(spec_shape[0])):
  #         for frame in range(frame_nums):
  #             # pow_frames = ((1.0 / frame_length) * ((spectrogram[frame,:,:]) ** 2))
  #             # pow_frames = tf.cast((spectrogram[frame, :, :]) ** 2,dtype=tf.float64)
  #             pow_frames = tf.cast((spectrogram[i,:,frame, :]) ** 2, dtype=tf.float64)
  #
  #             low_freq_mel = 0
  #             high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
  #             # 将Hz转换为Mel
  #             mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
  #             # 使得Mel scale间距相等
  #             hz_points = (700 * (10 ** (mel_points / 2595) - 1))
  #             # 将Mel转换为Hz
  #             bin = np.floor((frame_length + 1) * hz_points / sample_rate)
  #             fbank = np.zeros((nfilt, int(np.floor(int(spec_shape[-1])))))
  #             for m in range(1, nfilt + 1):
  #                 f_m_minus = int(bin[m - 1])  # 左
  #                 f_m = int(bin[m])  # 中
  #                 f_m_plus = int(bin[m + 1])  # 右
  #
  #                 for k in range(f_m_minus, f_m):
  #                     fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
  #                 for k in range(f_m, f_m_plus):
  #                     fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
  #             #pow_frames = tf.reshape(pow_frames, (1, int(pow_frames.shape[0])))
  #             filter_banks = tf.matmul(pow_frames, fbank.T)
  #             filter_banks = tf.where(filter_banks == 0, np.full((filter_banks.shape), np.finfo(float).eps),
  #                                     filter_banks)  # 数值稳定性
  #             filter_banks = tf.cast(filter_banks, dtype=tf.float32)
  #             filter_banks = 20 * tf.math.log(filter_banks) / tf.math.log(10.0)  # dB 有错
  #             filter_banks -= (tf.reduce_mean(filter_banks, axis=1) + 1e-8)  # todo 改到这里
  #             # 保留所得到的倒频谱系数2-13
  #             mfcc = tf.signal.dct(filter_banks, type=2, axis=-1, norm='ortho')[:, 1: (num_ceps + 1)]
  #             # 查看filter_bank   mfcc的不同shape
  #             # 归一化倒谱提升窗口
  #             mfcc *= lifts
  #             if frame == 0:
  #                 mfcc_matrix = mfcc
  #             else:
  #                 mfcc_matrix = tf.concat([mfcc_matrix, mfcc], axis=0)
  #         #mfcc_matrix = tf.transpose(mfcc_matrix, perm=[1, 0])
  #         mfcc_matrix = tf.expand_dims(mfcc_matrix//100,axis=0)
  #         if i == 0:
  #             mel = mfcc_matrix
  #         else:
  #             mel = tf.concat([mel, mfcc_matrix], axis=0)
  #     mel = tf.expand_dims(mel,axis=1)
  #
  #     return mel

