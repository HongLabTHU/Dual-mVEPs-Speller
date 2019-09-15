import itertools
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mne import baseline
from scipy.ndimage import convolve1d
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, \
    precision_recall_curve, average_precision_score, roc_auc_score

characters = list(range(65, 91)) + list(range(48, 58))


def cls2target(cls, evt, total_evt=12):
    """
    return row/col index given cls and evt,
    used in bidirectional mode
    :param cls: classifier decisions, n-d array, last axis the class
    :param evt: evt order, n-d array last axis the evt index\
    :param total_evt: int default 12
    :return: row/col index n-d array
    """
    # col evt to
    evt = evt.copy() % (total_evt // 2)
    left_ind = cls == 1
    right_ind = cls == 2
    indices = np.zeros_like(cls)
    if left_ind.size:
        indices[left_ind] = evt[left_ind]
    if right_ind.size:
        indices[right_ind] = (evt[right_ind] + total_evt // 4) % (total_evt // 2)
    return indices


def softmax(z, copy=True):
    """
    Softmax function for one sample
    :param z: scores
    :param copy: if copy the array
    :return:
    """
    if copy:
        z = z.copy()
    max_prob = np.max(z)
    z -= max_prob
    z = np.exp(z)
    z /= np.sum(z)
    return z


def char2index(c):
    return ord(c) - ord('A') if ord(c) >= 65 else ord(c) - ord('0') + 26


def index2char(index):
    return chr(characters[index])


def find_nearest_time(subj_path):
    """
    Looking for nearest date in subject data directory
    :param subj_path: <path to subject data>
    :return:
    """

    def valid_time(time):
        try:
            date = datetime.strptime(time, "%Y-%m-%d-%H-%M-%S")
            return date
        except ValueError:
            return None

    subdir = os.listdir(subj_path)
    # str to datetime
    datetimes = list(map(valid_time, subdir))
    # filter out none
    datetimes = list(filter(lambda x: x is not None, datetimes))
    datetimes.sort()
    # nearest
    nearest = datetimes[-1]
    # datetime to str
    return nearest.strftime("%Y-%m-%d-%H-%M-%S")


def cut_epochs(t, data, timestamps):
    """
    cutting raw data into epochs
    :param t: tuple (start, end, samplerate)
    :param data: ndarray (n_channels, n_times)
    :param timestamps: list of timestamps
    :return: None or ndarray (n_epochs, n_channels, n_times)
    """
    assert data.ndim == 2
    timestamps = np.array(timestamps)
    start = timestamps + int(t[0] * t[2])
    end = timestamps + int(t[1] * t[2])
    epochs = np.stack([data[:, s:e] for s, e in zip(start, end)], axis=0)
    return epochs


def sort_epochs(epochs, event):
    """
    sorting epoch data according to event order
    :param epochs: 3d epochs (n_epochs, n_channels, time_seq)
    :param event: 2d event array (n_trials, 12)
    :return:
        sorted_epochs: ndarray, with the same shape of "epochs"
    """
    assert epochs.ndim == 3
    rep_dim = event.shape[1]
    indices = np.argsort(event, axis=-1).flatten()
    for i in range(0, indices.shape[0], rep_dim):
        indices[i:i + rep_dim] += i
    sorted_epochs = epochs[indices]  # deep copy
    return sorted_epochs


# apply_baseline use mne.
def apply_baseline(t, data, mode='mean'):
    """
    Simple wrapper of mne rescale function
    :param t: tuple (start, end, samplerate)
    :param data: ndarray of any shape with axis=-1 the time axis
    :param mode: 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        refer to mne.baseline.rescale
    :return: ndarray
    """
    start, end, samplerate = t
    base = (start, 0)
    times = np.linspace(start, end, data.shape[-1])
    data = baseline.rescale(data, times, baseline=base, mode=mode, verbose=False)
    return data


def chan_select(x, label, n_best):
    """
    channel selection based on mutual information.
    :param x: 3D epoch data (n_epochs, n_channels, timesteps)
    :param label:
    :param n_best: channels to select, set to -1 to select all
    :return:
    """
    if x.shape[1] > n_best > 0:
        scores = chan_mut_info(x, label)
        ind = np.argsort(scores)[-n_best:].tolist()
        # ind must be sorted, or the model would be trained with wrong feature order
        ind.sort()
        return ind
    else:
        # select all
        return list(range(x.shape[1]))


def average(data, n=3, axis=0):
    """
    take average of epoch data
    :param data: ndarray, the first dimension the epoch count
    :param n: average n epochs
    :param axis: axis to take average
    :return:
    """
    kernel = np.ones((n,)) / n
    mean_resp = convolve1d(data, kernel, axis=axis, mode='reflect')
    return mean_resp


def average_multiclass(data, labels, n=3):
    """
    take average for multiclass epoch data
    :param data:
    :param labels:
    :param n:
    :return:
    """
    # map labels
    uni_label = np.unique(labels)
    book_list = [labels == i for i in uni_label]
    x_list = [data[book] for book in book_list]

    # map average function
    data_list = list(map(lambda x: average(x, n), x_list))
    # reshape
    mean_resp = np.zeros_like(data)
    for i, book in enumerate(book_list):
        mean_resp[book] = data_list[i]
    return mean_resp


def get_label(stim_string, n_rep):
    keyboard = {name: i for i, name in enumerate(characters)}
    assert isinstance(n_rep, int) and n_rep > 0

    label = np.zeros((len(stim_string), 12), dtype=np.int32)
    index = [keyboard[ord(char)] for char in stim_string]

    row_order = list(map(lambda x: x // 6, index))
    col_order = list(map(lambda x: x % 6 + 6, index))

    x = np.arange(len(stim_string))
    label[x, row_order] = 1
    label[x, col_order] = 1

    # tile
    label = np.tile(label[:, None, :], (1, n_rep, 1)).reshape((-1, 12))

    return label


def get_label_bidir(stim_string, n_rep, events):
    # TODO: optimize and get rid of the for loops
    keyboard = np.reshape(characters, (6, 6))
    st_order = np.sort(events, axis=-1)

    def is_target(num, order):
        assert 0 <= order < 12
        if order < 6:
            return num in keyboard[order]
        else:
            return num in keyboard[:, order - 6]

    string = []
    for c in stim_string:
        string.extend([c] * n_rep)
    label = []
    for i, c in enumerate(string):
        num = ord(c)
        for order in st_order[i]:
            if order in {3, 4, 5, 9, 10, 11}:
                left = is_target(num, order)
                right = is_target(num, order - 3)
            else:
                left = is_target(num, order)
                right = is_target(num, order + 3)
            if not left and not right:
                label.append(0)
            elif left:
                label.append(1)
            elif right:
                label.append(2)
            else:
                raise ValueError('Left and right are both target: impossible')
    return np.array(label)


def timewindow(t_ori, t_target, epochs):
    """
    Cut epoch data according to the provided time window
    :param t_ori: tuple, original time window (start, end)
    :param t_target: tuple, target time window
    :param epochs: 3d array
    :return:
    """
    assert t_target[0] >= t_ori[0] and t_target[1] <= t_ori[1], print(t_target, t_ori)
    start, end = t_ori
    unit = epochs.shape[-1] / (end - start)
    ind_s, ind_e = int((t_target[0] - start) * unit), int((t_target[1] - start) * unit)
    return epochs[..., ind_s:ind_e]


def pred_ave(classifier, data, labels, n=3):
    """
    Calculate mean confidence of inputs (only used in offline procedure)
    :param classifier: trained model, with decision_function() attribute
    :param data: (n_samples, n_dims)
    :param labels: (n_samples, )
    :param n: average time
    :return: mean confidence
    """
    assert hasattr(classifier, "decision_function")
    scores = classifier.decision_function(data)
    scores = average_multiclass(scores, labels, n=n)
    return scores


def chan_mut_info(epochs, labels):
    """
    Calculate mean channel mutual information scores, used for channel selection
    :param epochs:
    :param labels:
    :return:
    """

    def chan_mut_info_binary(epochs, labels):
        shape = epochs.shape
        x = epochs.reshape((shape[0], -1))
        scores = mutual_info_classif(x, labels)
        scores = scores.reshape(shape[1:]).mean(axis=-1)
        return scores

    cls_ind = np.unique(labels)
    if cls_ind.size == 2:
        scores = chan_mut_info_binary(epochs, labels)
    else:
        # discrete to one hot
        label_one_hot = np.eye(len(cls_ind))[labels]
        scores = []
        for i in range(label_one_hot.shape[-1]):
            score = chan_mut_info_binary(epochs, label_one_hot[:, i])
            scores.append(score)
        # mean scores over all classes
        scores = np.mean(scores, axis=0)
    return scores


def uniform_kfold(y, k=10, shuffle=True, random_state=52):
    """
    give uniform samples for each class when performing k fold validation
    :param y: label
    :param k: k-fold
    :param shuffle: if shuffle data
    :param random_state
    :return: generator of k fold indices
    """
    y = np.array(y)
    classes = np.unique(y)
    classes_list = [np.nonzero(y == i)[0] for i in classes]
    if shuffle:
        np.random.seed(random_state)
        for cls in classes_list:
            np.random.shuffle(cls)
    # yield k fold indices
    for i in range(k):
        test_start = [int(len(cls) / k * i) for cls in classes_list]
        test_end = [int(len(cls) / k * (i + 1)) for cls in classes_list]

        test_ind = []
        for cls, s, e in zip(classes_list, test_start, test_end):
            test_ind.extend(cls[s:e])
        train_ind = []
        for cls, s, e in zip(classes_list, test_start, test_end):
            train_ind.extend(cls[:s])
            train_ind.extend((cls[e:]))
        yield train_ind, test_ind


def uniform_split(y, split=0.2, shuffle=True, random_state=52):
    """

    :param y: label
    :param split: split ratio for validation
    :param shuffle: shuffle label?
    :param random_state: set random seed to reproduce result
    :return:
    """
    y = np.array(y)
    classes = np.unique(y)
    classes_list = [np.nonzero(y == i)[0] for i in classes]
    if shuffle:
        np.random.seed(random_state)
        for cls in classes_list:
            np.random.shuffle(cls)
    mid = [int(len(cls) * split) for cls in classes_list]
    test_ind = []
    train_ind = []
    for cls, m in zip(classes_list, mid):
        test_ind.extend(cls[:m])
        train_ind.extend(cls[m:])
    return train_ind, test_ind


def estimate_accu_uni(y_true, y_pred, n_avg=3):
    # y for non-target
    y_n = y_pred[y_true == 0]
    # y for target
    y_t = y_pred[y_true == 1]
    # estimate the parameter, maximum-likelihood estimation
    mu = np.array([np.mean(y_i) for y_i in [y_n, y_t]])
    sigma = np.array([np.std(y_i) for y_i in [y_n, y_t]])
    pc_avg = []
    for i in range(1, n_avg + 1):
        sigma_avg = sigma / np.sqrt(i)
        x = np.arange(y_pred.min(), y_pred.max(), 0.01)
        t_pdf = stats.norm.pdf(x, loc=mu[1], scale=sigma_avg[1])
        n_cdf = stats.norm.cdf(x, loc=mu[0], scale=sigma_avg[0])
        pc = np.trapz(t_pdf * n_cdf ** 5, x) ** 2
        pc_avg.append(pc)
    return pc_avg


def estimate_accu_dual(y_true, y_pred, n_avg=3):
    # non-target
    y_nl = y_pred[y_true == 0, 1]
    y_nr = y_pred[y_true == 0, 2]
    # leftward
    y_ll = y_pred[y_true == 1, 1]
    y_lr = y_pred[y_true == 1, 2]
    # rightward
    y_rl = y_pred[y_true == 2, 1]
    y_rr = y_pred[y_true == 2, 2]
    y = [y_nl, y_nr, y_ll, y_lr, y_rl, y_rr]
    # estimate the parameter, maximum-likelihood estimation
    mu = np.array([np.mean(y_i) for y_i in y])
    sigma = np.array([np.std(y_i) for y_i in y])
    pc_avg = []
    for i in range(1, n_avg + 1):
        sigma_avg = sigma / np.sqrt(i)
        x = np.arange(y_pred[:, 1:].min(), y_pred[:, 1:].max(), 0.01)
        # target: theta
        t_pdf_l = stats.norm.pdf(x, loc=mu[2], scale=sigma_avg[2])
        t_pdf_r = stats.norm.pdf(x, loc=mu[5], scale=sigma_avg[5])
        # non-target: gamma
        n_cdf_lr = stats.norm.cdf(x, loc=mu[3], scale=sigma_avg[3])
        n_cdf_nl = stats.norm.cdf(x, loc=mu[0], scale=sigma_avg[0])
        n_cdf_nr = stats.norm.cdf(x, loc=mu[1], scale=sigma_avg[1])
        n_cdf_rl = stats.norm.cdf(x, loc=mu[4], scale=sigma_avg[4])
        # pl and pr
        n_cdf_n = n_cdf_nl ** 2 * n_cdf_nr ** 2
        pl = np.trapz(t_pdf_l * n_cdf_lr * n_cdf_n, x)
        pr = np.trapz(t_pdf_r * n_cdf_rl * n_cdf_n, x)
        pc = ((pl + pr) / 2) ** 2
        pc_avg.append(pc)
    return pc_avg


def evaluate_binary(_y, y_true, cls_name=('non-target', 'target'), if_plot=True):
    y_pred_label = _y > 0.5
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred_label, average='binary')
    fpr, tpr, _ = roc_curve(y_true, _y)
    accu = accuracy_score(y_true, y_pred_label)
    conf = confusion_matrix(y_true, y_pred_label)
    _precision, _recall, _ = precision_recall_curve(y_true, _y)
    if if_plot:
        # plot roc
        fig_roc, auc = plot_roc_curve([(fpr, tpr)], cls_name=cls_name)
        # plot confusion matrix
        fig_cm, _ = plot_confusion_matrix(conf, classes=cls_name, normalize=True)
        # plot precision recall curve
        fig_pr, ap = plot_pr_curve([(_precision, _recall)], cls_name=cls_name)
    else:
        auc = np.trapz(tpr, fpr)
        ap = - np.trapz(_precision, _recall)

    # param dict
    result = {
        'accuracy': accu,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'AUC': auc,
        'mAP': ap
    }
    if if_plot:
        return fig_roc, fig_cm, fig_pr, result
    else:
        return result


def evaluate_multiclass(_y, y_true, cls_name=('non-target', 'left', 'right'), if_plot=True):
    n_classes = len(cls_name)
    assert n_classes > 2
    assert _y.shape[1] == n_classes == len(np.unique(y_true))

    y_pred_label = np.argmax(_y, axis=1)
    # onehot label
    y_true_onehot = np.eye(n_classes)[y_true]
    # confusion matrix
    conf = confusion_matrix(y_true, y_pred_label)
    if if_plot:
        fig_cm, conf = plot_confusion_matrix(conf, normalize=True, classes=cls_name)
    else:
        # normalize confusion matrix
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    # mAP & roc
    if if_plot:
        ap_curve = []
        auc_curve = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], _y[:, i])
            ap_curve.append((precision, recall))
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], _y[:, i])
            auc_curve.append((fpr, tpr))

        fig_pr, ap = plot_pr_curve(ap_curve, cls_name)
        fig_roc, auc = plot_roc_curve(auc_curve, cls_name)
    else:
        ap = average_precision_score(y_true_onehot, _y)
        auc = roc_auc_score(y_true_onehot, _y)
    result = {
        'AUC': auc,
        'mAP': ap,
        'mean accuracy': np.trace(conf) / n_classes
    }
    if if_plot:
        return fig_roc, fig_cm, fig_pr, result
    else:
        return result


def itr(p, n, t):
    """
    information transfer rate in bits/min
    """
    trans = np.log2(n)
    if 0 < p < 1:
        loss = p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    elif p == 0:
        loss = -trans
    else:
        loss = 0
    it = trans + loss
    return it / t * 60


########################################################################################################################
# utils for visualization
def draw_average(t, epoch, label, ch_names=None, cls_name=('nontarget', 'target'), with_error=True, **kwargs):
    """
    Draw average response
    :param t: tuple (start, end, samplerate)
    :param epoch: (epochs, channels, time)
    :param ch_names: list names of selected channels
    :param cls_name: list of classes name
    :param label: label (epochs, )
    :param with_error: whether or not to draw error shadow
    :param fig_size: figure size. default (15, 15)
    :return: plt.figure object
    """
    start, end, fs = t
    t = np.linspace(start, end, epoch.shape[-1])
    unique_class = np.unique(label)
    cls_list = list(map(lambda i: epoch[label == i], unique_class))

    if ch_names is None:
        ch_names = [str(i) for i in range(epoch.shape[1])]

    if with_error:
        e_list = list(map(lambda x: np.std(x, axis=0) / np.sqrt(x.shape[0]), cls_list))

    ave_cls_list = list(map(lambda x: np.mean(x, axis=0), cls_list))
    if epoch.shape[1] == 1:
        fig, ax = plt.subplots(1, 1, **kwargs)
        for ave, cls in zip(ave_cls_list, cls_name):
            ax.plot(t, ave[0], linestyle='-', label=cls)

        if with_error:
            for ave, e_ave in zip(ave_cls_list, e_list):
                ax.fill_between(t, ave[0] - e_ave[0], ave[0] + e_ave[0], alpha=0.1)

        ax.legend()
        ax.set_title(ch_names[0], fontsize=14)
        ax.tick_params('both', labelsize='large')
        ax.set_xticks([t[0], 0, t[-1]])
        ax.set_xticklabels(['%.2f' % t[0], '0', '%.2f' % t[-1]])
        ax.locator_params(axis='y', nbins=6)
    else:
        fig, axes = plt.subplots(int(math.ceil(len(ch_names) / 2)), 2, **kwargs)
        for i, ax in enumerate(axes.flat):
            if i < len(ch_names):
                for ave, cls in zip(ave_cls_list, cls_name):
                    ax.plot(t, ave[i], linestyle='-', label=cls)

                if with_error:
                    for ave, e_ave in zip(ave_cls_list, e_list):
                        ax.fill_between(t, ave[i] - e_ave[i], ave[i] + e_ave[i], alpha=0.1)

                ax.legend()
                ax.set_title(ch_names[i], fontsize=14)
                ax.tick_params('both', labelsize='large')
                ax.set_xticks([t[0], 0, t[-1]])
                ax.set_xticklabels(['%.2f' % t[0], '0', '%.2f' % t[-1]])
                ax.locator_params(axis='y', nbins=6)
    return fig


def draw_trial_im(t, epoch, label, n_avg=1, ch_names=None, cls_name=('non-target', 'target'), title='', **kwargs):
    """
    Plot epochs trial by trial in an image.
    :param t: tuple (start, end, samplerate)
    :param epoch: (epochs, channels, time)
    :param label:
    :param n_avg:
    :param ch_names:
    :param cls_name:
    :param kwargs: plt.figure arguments
    :return: figs: list of figure (one for each class)
    """
    start, end, fs = t
    unique_class = np.unique(label)
    # apply sliding window average
    if n_avg > 1:
        epoch = average_multiclass(epoch, label, n=n_avg)
    else:
        epoch = epoch.copy()
    epoch -= epoch.mean(axis=(0, 2), keepdims=True)
    epoch /= epoch.std(axis=(0, 2), keepdims=True)

    cls_list = list(map(lambda i: epoch[label == i], unique_class))
    fig_list = []
    for epochs_cls, name in zip(cls_list, cls_name):
        if len(ch_names) > 1:
            fig, axes = plt.subplots(int(math.ceil(len(ch_names) / 2)), 2, **kwargs)
        else:
            fig, axes = plt.subplots(1, 1, **kwargs)
            axes = np.array([axes])
        fig_list.append(fig)
        fig.suptitle(title + ' ' + name)
        # randomly choose 120 samples
        idx = np.random.choice(120, size=(120,), replace=False)
        epoch_subsample = epochs_cls[idx]
        for i, ax in enumerate(axes.flat):
            if i < len(ch_names):
                im = ax.imshow(epoch_subsample[:, i],
                               cmap='RdBu_r',
                               aspect='auto',
                               extent=(start, end, 0, 120),
                               vmin=-3,
                               vmax=3)
                ax.set_title(ch_names[i])
        fig.colorbar(im, ax=axes.ravel().tolist())
    return fig_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig, cm


def plot_roc_curve(data, cls_name, title='ROC curve'):
    """

    :param data: list [(fpr, tpr), (), ...]
    :param cls_name: tuple of names for each class
    :param title: plot title
    :return:
    """

    def cal_auc(tpr, fpr):
        return np.trapz(tpr, fpr)

    def plot_single_curve(fpr, tpr, cls_ind):
        auc = cal_auc(tpr, fpr)
        plt.plot(fpr, tpr, label="%s ROC curve (area = %.2f)" % (cls_name[cls_ind], auc))
        return auc

    assert isinstance(data, list)
    if len(cls_name) == 2:
        assert len(data) == 1
    else:
        assert len(data) == len(cls_name)

    fig = plt.figure()

    args = [(fpr, tpr, i) for i, (fpr, tpr) in enumerate(data)]

    if len(cls_name) > 2:
        auc = np.mean(list(map(lambda x: plot_single_curve(*x), args)))
    else:
        fpr, tpr = data[0]
        auc = cal_auc(tpr, fpr)
        plt.plot(fpr, tpr, label="%s vs. %s ROC curve (area = %.2f)" % (cls_name[1], cls_name[0], auc))

    ax = plt.gca()
    ax.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.title(title + ' (mean area = %.4f)' % auc)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    return fig, auc


def plot_pr_curve(data, cls_name, title='Precision Recall Curve'):
    """

    :param data: list [(precision, recall), (), ...]
    :param cls_name: tuple of names for each class
    :param title: plot title
    :return:
    """

    def cal_ap(precision, recall):
        return -np.trapz(precision, recall)

    def plot_single_curve(precision, recall, cls_ind):
        ap = cal_ap(precision, recall)
        plt.plot(recall, precision, label="%s PR curve (area = %.2f)" % (cls_name[cls_ind], ap))
        return ap

    assert isinstance(data, list)
    if len(cls_name) == 2:
        assert len(data) == 1
    else:
        assert len(data) == len(cls_name)

    fig = plt.figure()

    args = [(precision, recall, i) for i, (precision, recall) in enumerate(data)]

    if len(cls_name) > 2:
        ap = np.mean(list(map(lambda x: plot_single_curve(*x), args)))
    else:
        precision, recall = data[0]
        ap = cal_ap(precision, recall)
        plt.plot(recall, precision, label="%s vs. %s PR curve (area = %.2f)" % (cls_name[1], cls_name[0], ap))

    plt.title(title + ' (mean area = %.4f)' % ap)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    return fig, ap
