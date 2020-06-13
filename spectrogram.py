#!/usr/bin/env python
# coding: utf8

""" Spectrogram specific data augmentation """

# pylint: disable=import-error
import numpy as np
import tensorflow as tf
import math
import librosa
import pylab as plt
import librosa.display
import pickle
#from tensorflow.signal import stft, hann_window
from scipy.fftpack import dct
# pylint: enable=import-error


__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def compute_spectrogram_tf(
        waveform,
        frame_length=2048, frame_step=512,
        spec_exponent=1., window_exponent=1.):
    """ Compute magnitude / power spectrogram from waveform as
    a n_samples x n_channels tensor.

    :param waveform:        Input waveform as (times x number of channels)
                            tensor.
    :param frame_length:    Length of a STFT frame to use.
    :param frame_step:      HOP between successive frames.
    :param spec_exponent:   Exponent of the spectrogram (usually 1 for
                            magnitude spectrogram, or 2 for power spectrogram).
    :param window_exponent: Exponent applied to the Hann windowing function
                            (may be useful for making perfect STFT/iSTFT
                            reconstruction).
    :returns:   Computed magnitude / power spectrogram as a
                (T x F x n_channels) tensor.
    """
    waveform = tf.constant(waveform,dtype=waveform.dtype)
    stft_tensor = tf.transpose(
        tf.signal.stft(
            tf.transpose(waveform),
            frame_length,
            frame_step,
            window_fn=lambda f, dtype: tf.signal.hann_window(
                f,
                periodic=True,
                dtype=waveform.dtype) ** window_exponent),
        #perm=[1, 2, 0])
        perm=[0,1])
    return np.abs(stft_tensor) ** spec_exponent
    #return np.abs(stft_tensor) ** spec_exponent, stft_tensor



def compute_mfcc_tf(spectrogram,
        frame_length=2048, sample_rate=44100,
        nfilt=390,
        num_ceps=128):
    spec_shape = spectrogram.shape
    frame_nums = spec_shape[0]
    lifts = []
    for n in range(1, num_ceps+1):
        lift = 1 + math.floor(num_ceps) * np.sin(np.pi * n / 12)
        lifts.append(lift)
    #print(lifts)
    for frame in range(frame_nums):
        #pow_frames = ((1.0 / frame_length) * ((spectrogram[frame,:,:]) ** 2))
        #pow_frames = tf.cast((spectrogram[frame, :, :]) ** 2,dtype=tf.float64)
        pow_frames = tf.cast((spectrogram[frame, :]) ** 2, dtype=tf.float64)

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        # 将Hz转换为Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        # 使得Mel scale间距相等
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        # 将Mel转换为Hz
        bin = np.floor((frame_length + 1) * hz_points / sample_rate)
        fbank = np.zeros((nfilt, int(np.floor(int(spec_shape[1])))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # 左
            f_m = int(bin[m])  # 中
            f_m_plus = int(bin[m + 1])  # 右

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        pow_frames = tf.reshape(pow_frames,(1,int(pow_frames.shape[0])))
        filter_banks = tf.matmul(pow_frames,fbank.T)
        filter_banks = tf.where(filter_banks == 0, np.full((filter_banks.shape), np.finfo(float).eps), filter_banks)# 数值稳定性
        filter_banks = tf.cast(filter_banks,dtype=tf.float32)
        filter_banks = 20 * tf.math.log(filter_banks)/tf.math.log(10.0)  # dB 有错
        filter_banks -= (tf.reduce_mean(filter_banks, axis=1) + 1e-8)# todo 改到这里
        # 保留所得到的倒频谱系数2-13
        mfcc = tf.signal.dct(filter_banks, type=2, axis=-1, norm='ortho')[:, 1: (num_ceps + 1)]
        # 查看filter_bank   mfcc的不同shape
        # 归一化倒谱提升窗口
        mfcc *= lifts
        if frame == 0:
            mfcc_matrix = mfcc
        else:
            mfcc_matrix = tf.concat([mfcc_matrix,mfcc],axis=0)
    mfcc_matrix = tf.transpose(mfcc_matrix,perm=[1,0])
    return mfcc_matrix





def get_mfcc(waveform,
             sample_rate):
    # 画出波形图
    librosa.display.waveplot(waveform, sr=sample_rate)
    # 提取MFCC Returns  M: np.ndarray [shape=(n_mfcc, t)]
    # MFCC sequence
    mfccs = librosa.feature.melspectrogram(waveform, sr=sample_rate)
    # 获取特征值的维度
    print('------------------------------')
    print(mfccs.shape)  # 打印将输出(n_mfcc, t))
    # 画出MFCC的图（上方第二个图）
    #librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    # 对MFCC的数据进行处理
    #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    # 画出处理后的图（上方第三个图）
    #librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    #plt.plot(mfccs)
    #plt.show()
    return mfccs


#sess = tf.Session()
waveform ,fs = librosa.load('/mnt/hgfs/share/1.wav',sr=16000)
spectrogram = compute_spectrogram_tf(waveform,
         frame_length=2048, frame_step=512,
         spec_exponent=1., window_exponent=1.)
a = compute_mfcc_tf(spectrogram,
        frame_length=1024, sample_rate=16000,
        nfilt=390,
        num_ceps=128)

waveform, fs = librosa.load(r'/mnt/hgfs/share/1.wav',sr=16000)
b = get_mfcc(waveform,fs)
w = 1

# spectrogram = sess.run(compute_spectrogram_tf(
#         waveform,
#         frame_length=2048, frame_step=512,
#         spec_exponent=1., window_exponent=1.))