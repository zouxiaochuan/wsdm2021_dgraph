

import numpy as np
import constants


def get_label(
        current_ts, target_ts, bin_size, neg_interval, neg_num):
    current_bin = current_ts // bin_size
    target_bins = target_ts // bin_size

    target_bins = np.append(target_bins, current_bin)
    
    max_target_ts = max(target_ts[-1], current_ts + neg_interval)
    
    target_bin_range = np.array(
        [current_ts - neg_interval, max_target_ts],
        dtype='int32') // bin_size

    target_bin_range[1] += 1

    neg_bins = np.arange(target_bin_range[0], target_bin_range[1])
    neg_labels = np.zeros(len(neg_bins), dtype='float32')
    neg_labels[target_bins - neg_bins[0]] = 1
    # neg_labels[current_bin - neg_bins[0]] = 1
    # neg_bins = np.setdiff1d(neg_bins, target_bins)

    if len(neg_bins) > neg_num:
        selected_bins = np.random.choice(
            np.arange(len(neg_bins)),
            size=neg_num, replace=False)

        neg_bins = neg_bins[selected_bins]
        neg_labels = neg_labels[selected_bins]
        pass

    label_bins = np.insert(neg_bins, 0, current_bin)
    label = np.insert(neg_labels, 0, 1)

    return label_bins, label


def get_predict_bins(start_ts, end_ts, bin_size):
    start_bin = start_ts // bin_size
    end_bin = end_ts // bin_size + 1

    bins = np.arange(start_bin, end_bin)
    weights = np.ones(len(bins))

    if len(bins) > 1:
        weights[0] = ((start_bin + 1) * bin_size - start_ts) / bin_size
        weights[-1] = (end_ts - bins[-1] * bin_size) / bin_size
        pass
    else:
        weights[0] = 1
        pass

    return bins, weights
    pass
