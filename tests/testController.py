import unittest
import shutil
from queue import Queue

import numpy as np

from Offline.dataset import Dataset
from Offline import utils
from config import cfg
from Online.Controller import TestingController


class _TestDataclient:


    def __init__(self, dataset, fs):
        self.epochs_per_trial = 6  if cfg.exp_config.bidir else 12  # dual-directional paradigm -> 6 uni- -> 12
        self.data = dataset.data[dataset.montage_indices]
        self.timestamps = np.array(dataset.timestamp)
        self.events = dataset.events
        self._t_addon = int(fs * 0.6)
        self._ts_index = 0 + self.epochs_per_trial * 2
        self._char_cnt = 0

    def get_trial_data(self):
        s = max(self.timestamps[self._ts_index] - self._t_addon, 0)
        e = min(self.timestamps[self._ts_index + self.epochs_per_trial - 1] + self._t_addon, self.data.shape[-1])
        raw_data = self.data[:, s:e]
        timestamps = self.timestamps[self._ts_index:self._ts_index + self.epochs_per_trial] - s
        self._ts_index += self.epochs_per_trial
        return timestamps.tolist(), raw_data

    def get_to_next_char(self):
        self._char_cnt += 1
        self._ts_index = self.epochs_per_trial * 10 * self._char_cnt + self.epochs_per_trial * 2

    def get_trial_events(self):
        def build_dual_events(event):
            if event > 6:
                event2 = ((event - 6) + 3) % 6 + 6
            else:
                event2 = (event + 3) % 6
            return (event, event2)
        evt_ind = self._ts_index // self.epochs_per_trial
        if cfg.exp_config.bidir:
            events = list(map(build_dual_events, self.events[evt_ind]))
        else:
            events = self.events[evt_ind].tolist()
        return events


def main(subjname, data_date, model_date):
    dataset = Dataset(subjname, date=data_date)
    queue_stim = Queue()
    queue_result = Queue()
    dataclient = _TestDataclient(dataset, fs=cfg.amp_info.samplerate)
    controller = TestingController(q_stim=queue_stim, q_result=queue_result, dataclient=dataclient,
                                   model_date=model_date, stim_string=cfg.exp_config.test_string)
    # main process to test controller
    chr_result = []
    for i, c in enumerate(cfg.exp_config.test_string):
        result = -1
        while result < 0:
            # get events
            events = dataclient.get_trial_events()
            queue_stim.put(events)
            controller.run()
            result = queue_result.get()
        chr_result.append(utils.index2char(result))
        print('predicted: %s' % chr_result[-1])
        print('ground truth: %s' % c)
        print()
        dataclient.get_to_next_char()
    itr_tuple = controller.itr()
    controller.close()
    # clean up newly made dir
    shutil.rmtree(controller._path)
    return chr_result, itr_tuple


class TestController(unittest.TestCase):
    # Test both eeg and seeg situations!
    def test_eeg(self):
        subj = 'TEST'
        cfg.exp_config.bidir = True
        cfg.subj_info.subjname = subj
        cfg.exp_config.test_string = 'AHOV29FKPUZ4'
        cfg.amp_info.samplerate = 1000
        cfg.off_config.k_best_channel = -1
        cfg.off_config.use_hg = False
        cfg.off_config.time_window = (0, 0.5)
        cfg.exp_config.smart_stopping = 0.95
        results, itr_tuple = main(subj, '2019-09-17-20-05-51', '2019-09-17-20-05-51')
        pred = np.array(results)
        target = np.array(list(cfg.exp_config.test_string))
        accu = np.mean((pred == target).astype(np.float64))
        print(accu)
        self.assertTrue(accu >= 1)
        print(itr_tuple)
        self.assertTrue(itr_tuple[0] == accu)
        self.assertTrue(itr_tuple[-1] >= 40)

    def test_decision_logic_bidir(self):
        cfg.subj_info.subjname = 'TEST'
        cfg.exp_config.bidir = True
        controller = TestingController(q_stim=None, q_result=None, dataclient=None,
                                       model_date=None, stim_string=cfg.exp_config.train_string)
        score = np.array([[-1, 0, 1.5], [0, 1, -1], [8, 0.4, -0.4], [2, 0.5, -0.5], [0, -0.5, 0.5], [0, -1, 1]])
        evt = np.array([0, 2, 1, 9, 10, 11])
        result, p = controller.decision_logic(score, evt)
        self.assertTrue(result == 20)
        # clean up newly made dir
        controller.close()
        shutil.rmtree(controller._path)

    def test_write_result(self):
        cfg.subj_info.subjname = 'test'
        cfg.exp_config.bidir = False
        controller = TestingController(q_stim=None, q_result=None, dataclient=None,
                                       model_date=None, stim_string=' a BC')
        controller._write_result_ind(0)
        with open(controller._log_path) as f:
            line = f.read()
        self.assertTrue(line == 'A A 0\n')
        controller.close()
        shutil.rmtree(controller._path)


if __name__ == '__main__':
    unittest.main()
