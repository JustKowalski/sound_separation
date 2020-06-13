# Created on 2018/12
# Author: Kaituo XU

from itertools import permutations


import numpy as np
import tensorflow as tf




EPS = 1e-8


def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


def cal_loss(source, estimate_source, source_lengths,max_snr=1e6, bias_ref_signal=None):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - tf.reduce_mean(max_snr)
    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.shape == estimate_source.shape
    B, C, T = source.shape
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = tf.reshape(source_lengths,shape=[-1,1,1])
    num_samples = tf.cast(num_samples,dtype=tf.float64)
    mean_target = tf.reduce_sum(source, axis=2, keepdims=True) / num_samples
    mean_estimate = tf.reduce_sum(estimate_source, axis=2, keepdims=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = tf.expand_dims(zero_mean_target, axis=1)
    s_estimate = tf.expand_dims(zero_mean_estimate, axis=2)
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = tf.reduce_sum(s_estimate * s_target, axis=3, keepdims=True)  # [B, C, C, 1]
    s_target_energy = tf.reduce_sum(s_target ** 2, axis=3, keepdims=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = tf.reduce_sum(pair_wise_proj ** 2, axis=3) / (tf.reduce_sum(e_noise ** 2, axis=3) + EPS)
    pair_wise_si_snr = 10 * (tf.math.log(pair_wise_si_snr + EPS)/tf.cast(tf.math.log(10.0),dtype=tf.float64))  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    # todo 确认一下数据类型
    height = len(range(C))
    temp = list(permutations(range(C)))
    perms = tf.reshape(temp,[len(temp),height])
    #perms = source.new_tensor(list(permutations(range(C))))
    # one-hot, [C!, C, C]
    #index = tf.expand_dims(perms, 2)
    index = perms
    perms_one_hot = tf.one_hot(indices=index,depth=3,axis=2)
    #perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)#todo onehot编码
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = tf.einsum('bij,pij->bp', pair_wise_si_snr, tf.cast(perms_one_hot,dtype=tf.float64))
    max_snr_idx = tf.argmax(snr_set, axis=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr = tf.reduce_max(snr_set, axis=1, keepdims=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.shape
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = tf_index_select(perms, dim=0, indices=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = np.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.shape
    mask = np.ones((B,1,T))
    #mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask




if __name__ == "__main__":
    B, C, T = 2, 3, 12
    # fake data
    # source = tf.random.uniform((B, C, T),4)
    # estimate_source = tf.random.uniform((B, C, T), 4)
    source = np.ones((B , C , T))
    estimate_source = np.ones( (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    temp = tf.zeros([T, T - 3],dtype=tf.float32)
    source_lengths = tf.shape(temp)
    #source_lengths = torch.LongTensor([T, T - 3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_len gths', source_lengths)

    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)
