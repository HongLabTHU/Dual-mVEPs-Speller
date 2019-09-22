import os
import unittest

import numpy as np
from scipy import signal
from mne import create_info, EpochsArray

import Offline.model as Model
import Offline.utils as util
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from config import merge_cfg_from_file

merge_cfg_from_file('./test_stimulator.yml')


class TestOfflineUtils(unittest.TestCase):

    def test_cut_epochs(self):
        timestamp = [1, 1.2, 2, 3]
        start, end, fs = -0.1, 0.5, 10
        timestamp = list(map(lambda x: int(x * fs), timestamp))
        data = np.zeros((1, fs * 4))
        start_p = tuple(map(lambda x: x + int(start * fs), timestamp))
        data[:, start_p] = 1
        t = (start, end, fs)
        epochs = util.cut_epochs(t, data, timestamp)
        self.assertTrue(
            epochs.shape == (4, 1, int(fs * (end - start))))
        self.assertTrue(np.allclose(epochs[:, :, 0], 1))

    def test_sort_epoch(self):
        n_epochs = 30
        x1 = np.arange(n_epochs)
        np.random.shuffle(x1)
        data = x1[:, None, None]
        event = np.arange(n_epochs)[None, :]
        sorted_epochs = util.sort_epochs(data, event)
        self.assertTrue(np.all(sorted_epochs.squeeze() == data.squeeze()))
        event = x1[None, :]
        sorted_epochs = util.sort_epochs(data, event)
        self.assertTrue(sorted_epochs.ndim == data.ndim)
        self.assertTrue(np.all(sorted_epochs.squeeze() == np.arange(n_epochs)))

        x2 = np.arange(n_epochs)
        np.random.shuffle(x2)
        data = np.concatenate((x1[:, None, None], x2[:, None, None]), axis=0)
        event = np.concatenate((x1[None, :], x2[None, :]), axis=0)
        sorted_epochs = util.sort_epochs(data, event)
        self.assertTrue(np.allclose(sorted_epochs.squeeze(), list(range(n_epochs)) * 2))

    def test_apply_baseline(self):
        sfreq = 10
        start, end = -0.1, 0.5
        t = np.linspace(start, end,
                        int(sfreq * (end - start)))
        data = np.array([np.sin(t)])
        baseline = data[..., :int(-start * sfreq)].mean(axis=-1, keepdims=True)
        tt = (start, end, sfreq)
        y = util.apply_baseline(tt, data)
        self.assertTrue(np.allclose(y, data - baseline))

    def test_feat_extractor(self):
        sfreq = 1200
        extractor = Model.FeatExtractor(sfreq=sfreq, band_erp=(1, 20))
        start, end = 0, 1.8
        t = np.linspace(start, end, int(sfreq * (end - start)), dtype=np.float64)
        assert len(t) > 3 * 200 / 5  # minimal sequence length! (len(seq) > len(padding) = 3 * max(b, a))
        # test erp
        ff = [3, 10, 30]
        x = np.array([np.sin(2 * np.pi * f * t) for f in ff])
        y = extractor(data=np.sum(x, axis=0))
        xx = x[:, int(sfreq * (start + 0.3)):int(sfreq * (end - 0.3))]
        yy = y[int(sfreq * (start + 0.3)):int(sfreq * (end - 0.3))]
        self.assertTrue(np.max(np.abs(yy - xx[0] - xx[1])) < 0.5)

        extractor = Model.FeatExtractor(sfreq=sfreq, band_erp=(1, 20))
        assert len(t) > 3 * 200 / 5  # minimal sequence length! (len(seq) > len(padding) = 3 * max(b, a))
        # test erp
        ff = [3, 10, 30]
        x = np.array([np.sin(2 * np.pi * f * t) for f in ff])
        y = extractor(data=np.sum(x, axis=0))
        self.assertTrue(np.allclose(y, x[0] + x[1], atol=0.3))

    def test_find_nearest_time(self):
        path = os.path.join(os.path.dirname(__file__), '../data/test')
        time = util.find_nearest_time(path)
        self.assertTrue(time == '2019-09-17-20-05-51')

    def test_average(self):
        n = 2
        data = np.zeros((3, 10, 1, 1))
        data[0, :] = 1
        data[1, :] = 0.5
        out = util.average(data, n)
        answer = np.array([0.75, 0.25, 0])
        self.assertTrue(np.allclose(out[:, 0].squeeze(), answer))

    def test_average_multiclass(self):
        data = np.zeros((5, 10))
        label = [-1, -1, 100, -1, 100]
        data[label == -1] = 1
        data[label == 100] = 100
        out = util.average_multiclass(data, label)
        self.assertTrue(np.allclose(out[label == -1], 1))
        self.assertTrue(np.allclose(out[label == 100], 100))

    def test_get_label(self):
        result_A = util.get_label('A', n_rep=1)
        self.assertTrue(np.allclose(result_A, [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]))
        result_0 = util.get_label('0', n_rep=1)
        self.assertTrue(np.allclose(result_0, [[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]))
        result_A00A = util.get_label('A00A', n_rep=1)
        self.assertTrue(np.allclose(result_A00A, [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]))
        result_A_3 = util.get_label('A0', n_rep=2)
        answer = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.int32)
        self.assertTrue(np.allclose(result_A_3, answer))

    def test_timewindow(self):
        data = np.arange(7)[None, None, :]
        t = (-0.1, 0.6)
        t_new = (0, 0.5)
        data_new = util.timewindow(t, t_new, data)
        self.assertTrue(np.all(data_new == np.arange(1, 6, 1)[None, None, :]))

    def test_pred_ave(self):
        data = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        label = np.array([-100, 100])
        classifier = LogisticRegression()
        classifier.fit(data, label)
        score = util.pred_ave(classifier, data, label, n=1)
        self.assertTrue(np.allclose(score > 0.5, [0, 1]))
        data = np.array([[0, 0, 0, 0],
                         [1, 1, 1, 1],
                         [0, 0, 0, 0],
                         [1, 1, 1, 1],
                         [0, 0, 0, 0],
                         [1, 1, 1, 1]])
        label = np.array([0, 1, 0, 1, 0, 1])
        score = util.pred_ave(classifier, data, label, n=3)
        self.assertTrue(np.allclose(score > 0.5, label))

    def test_k_fold(self):
        label = [4, 2, 2, 3, 2, 3, 4, 3]
        k_fold = [([2, 4, 5, 6, 7], [0, 1, 3]), ([0, 1, 3], [2, 4, 5, 6, 7])]
        for i, (train, test) in enumerate(util.uniform_kfold(label, k=2, shuffle=False)):
            train.sort()
            test.sort()
            self.assertTrue(np.allclose(train, k_fold[i][0]))
            self.assertTrue(np.allclose(test, k_fold[i][1]))
        k_fold = [([1, 2, 3, 5, 6], [0, 4, 7]), ([0, 4, 7], [1, 2, 3, 5, 6])]
        for i, (train, test) in enumerate(util.uniform_kfold(label, k=2, shuffle=True, random_state=0)):
            train.sort()
            test.sort()
            self.assertTrue(np.allclose(train, k_fold[i][0]))
            self.assertTrue(np.allclose(test, k_fold[i][1]))

    def test_draw_average(self):
        t = (-0.1, 0.5, 10)
        data = np.array([[[0, 1, 2, 3, 4, 5]],
                         [[-1, -2, -3, -4, -5, -6]],
                         [[-1, -2, -3, -4, -5, -6]],
                         [[-1, -2, 0, -4, -5, -6]]], dtype=np.float64)
        ch_names = ['test']
        label = np.array([0, 0, 1, 1])
        fig1 = util.draw_average(t, data, label, ch_names)
        plt.close()
        data = np.tile(data, (1, 2, 1))
        fig2 = util.draw_average(t, data, label, ['test1', 'test2'])
        plt.close()

    def test_confusion_matrix(self):
        cm = np.array([[999, 1], [20, 980]])
        util.plot_confusion_matrix(cm, classes=('a', 'b'), normalize=True)
        plt.close()
        self.assertTrue(True)

    def test_roc(self):
        fpr = np.linspace(0, 1, 1000)
        tpr = fpr ** 3
        util.plot_roc_curve([(fpr, tpr)], cls_name=('a', 'b'))
        plt.close()
        yy = - (fpr - 1) ** 2 + 1
        _, auc = util.plot_roc_curve([(fpr, tpr), (fpr, tpr), (fpr, yy)], cls_name=('c', 'b', 'a'))
        plt.close()
        self.assertTrue(np.allclose(auc, 7 / 18))

    def test_pr(self):
        recall = np.linspace(0, 1, 1000)
        recall = recall[::-1]
        precision = 1 - recall ** 3
        _, pr = util.plot_pr_curve([(precision, recall)], cls_name=('a', 'b'))
        plt.close()
        self.assertTrue(np.allclose(pr, 3 / 4))
        yy = - recall ** 2 + 1
        fig, pr = util.plot_pr_curve([(precision, recall), (precision, recall), (yy, recall)], cls_name=('c', 'b', 'a'))
        plt.close()
        self.assertTrue(np.allclose(pr, 13 / 18))

    def test_cls2target(self):
        cls = np.array([[2, 1], [1, 2], [1, 2]])
        evt = np.array([[3, 3], [0, 4], [1, 6]])
        ind = util.cls2target(cls, evt, total_evt=8)
        expected_result = np.array([[1, 3], [0, 2], [1, 0]])
        self.assertTrue(np.allclose(ind, expected_result))

    def test_chan_mut_info(self):
        # multiclass
        x = np.random.rand(3, 5, 5)
        y = np.arange(3)
        scores = util.chan_mut_info(x, y)
        self.assertTrue(scores.shape == (5,))
        # binary
        y = np.zeros((3,))
        y[-1] = 1
        scores = util.chan_mut_info(x, y)
        self.assertTrue(scores.shape == (5,))


if __name__ == '__main__':
    unittest.main()
