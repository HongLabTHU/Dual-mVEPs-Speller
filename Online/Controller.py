import os
from datetime import datetime
import json

import numpy as np

from Offline import utils
import queue
from Offline.model import Model, FeatExtractor
from config import cfg

quit_flag = False

word = utils.characters


def set_event_fio():
    # try to create subject folder in root/data
    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), '..', 'data', cfg.subj_info.subjname))
    except FileExistsError:
        print('Subject folder already exist')
    # create folder for this exp
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(os.path.dirname(__file__), '..', 'data', cfg.subj_info.subjname, time_str)
    os.mkdir(path)
    # create stim order file io
    return open(os.path.join(path, '%s.txt' % cfg.subj_info.subjname), 'w'), path


class Controller:
    def __init__(self, q_stim, q_result, ):
        self._fio, self._path = set_event_fio()
        self._q_stim = q_stim
        self._q_result = q_result
        self.total_trial_cnt = 0

    def _write_stim_order(self, stim_order):
        for stim in stim_order:
            if isinstance(stim, tuple):
                self._fio.write('%d %d ' % (stim[0], stim[1]))
            else:
                self._fio.write('%d ' % stim)
        self._fio.write('\n')

    def close(self):
        self._fio.close()

    def run(self):
        try:
            events = self._q_stim.get(block=False)
        except queue.Empty:
            return None

        self._write_stim_order(events)
        self.total_trial_cnt += 1
        return events

    def write_exp_log(self):
        # write exp info to log
        with open(os.path.join(self._path, 'exp_info.log'), 'w') as f:
            f.write(json.dumps(cfg.subj_info))
            f.write('\n')
            f.write(json.dumps(cfg.amp_info))
            f.write('\n')
            f.write(json.dumps(cfg.exp_config))


class TrainingController(Controller):
    def __init__(self, q_stim, q_result, stim_string):
        super(TrainingController, self).__init__(q_stim, q_result)

        self.rep_time = cfg.exp_config.n_rep
        self.stim_string = list(filter(lambda x: ord(x) in word, stim_string.upper()))
        # counting variables
        self.trial_cnt = 0
        self.char_cnt = 0

    def run(self):
        events = super(TrainingController, self).run()
        if events is None:
            return
        self.trial_cnt += 1
        if self.trial_cnt < self.rep_time:
            self._q_result.put(-1)
        else:
            result_index = utils.char2index(self.stim_string[self.char_cnt])
            self._q_result.put(result_index)
            self.trial_cnt = 0
            self.char_cnt += 1


class TestingController(Controller):
    total_evt = 12

    def __init__(self, q_stim, q_result, dataclient, stim_string, model_date=None):
        # find nearest path before creating new dataset folder
        self.root = os.path.join(os.path.dirname(__file__), '..', 'data', cfg.subj_info.subjname)
        if model_date is None:
            date = utils.find_nearest_time(self.root)
        else:
            date = model_date
        super(TestingController, self).__init__(q_stim, q_result)

        self.stim_string = list(filter(lambda x: ord(x) in word, stim_string.upper()))

        # another file to record decision results
        self._log_path = os.path.join(self._path, 'record.txt')
        # check if exist
        if os.path.isfile(self._log_path):
            raise FileExistsError("Record file already exists.")

        # error log file
        self._online_err_log = os.path.join(self._path, 'error.log')
        # check if exist
        if os.path.isfile(self._online_err_log):
            raise FileExistsError("Error log already exists.")

        # cnt variables
        self.trial_cnt = 0
        self.char_cnt = 0
        # trial cnt buffer
        self.trial_cnt_buf = []
        # char buffer
        self.result_buffer = []
        # load model
        try:
            self.model = Model(mode='test', date=date)
        except Exception as err:
            self.model = None
            print(err)
        # dataclient
        self.data_client = dataclient
        self.extractor = FeatExtractor(sfreq=cfg.amp_info.samplerate,
                                       band_erp=cfg.subj_info.erp_band)
        # score buffer
        if cfg.exp_config.bidir:
            self.score_buffer = np.zeros((self.total_evt // 2, 3), dtype=np.float64)
        else:
            self.score_buffer = np.zeros(self.total_evt, dtype=np.float64)

    def write_exp_log(self):
        # write exp info to log, testing controller need also to write offline configure info.
        with open(os.path.join(self._path, 'exp_info.log'), 'w') as f:
            f.write(json.dumps(cfg.subj_info))
            f.write('\n')
            f.write(json.dumps(cfg.amp_info))
            f.write('\n')
            f.write(json.dumps(cfg.exp_config))
            f.write('\n')
            f.write(json.dumps(cfg.off_config))

    def run(self):
        events = super(TestingController, self).run()
        if events is None:
            return
        self.trial_cnt += 1

        # process events, list to 2d array
        events = np.array([events])
        if cfg.exp_config.bidir:
            events = events[..., 0]

        timestamps, trial_data = self.data_client.get_trial_data()

        print(trial_data.shape, len(timestamps))

        # During online experiments, loss of trigger signal occurred occasionally.
        # This is because the wireless network transmission occasionally suffers from large interference,
        # which causes the trigger to be unable to synchronize with the data
        # collected by the amplifier due to excessive delay.
        # At this time, the online system would decide to abandon those trials and redo them.
        # We do not consider those trials when calculating ITR.
        if not timestamps:
            # write error log
            with open(self._online_err_log, 'a') as f:
                f.write('Do not receive any timestamps. Err Trial index: %d\n' % self.total_trial_cnt)
            self.trial_cnt -= 1
            self._q_result.put(-1)
            print('No timestamps. Trigger box disconnected!')
            return

        if timestamps[-1] + cfg.amp_info.samplerate * cfg.off_config.end > trial_data.shape[1]:
            # write error log
            with open(self._online_err_log, 'a') as f:
                f.write('Wrong data length. Err Trial index: %d\n' % self.total_trial_cnt)
            self.trial_cnt -= 1
            self._q_result.put(-1)
            print('Wrong data length')
            return

        if (len(timestamps) != 12 and not cfg.exp_config.bidir) or (len(timestamps) != 6 and cfg.exp_config.bidir):
            # write error log
            with open(self._online_err_log, 'a') as f:
                f.write('Wrong time stamp length. Err Trial index: %d\n' % self.total_trial_cnt)
            self.trial_cnt -= 1
            self._q_result.put(-1)
            print('Wrong time stamp length')
            return

        # process raw to extract features
        trial_feat = self.model.extract_feature(self.extractor, trial_data)
        # raw to epochs
        epochs = Model.raw2epoch(trial_feat, timestamps=timestamps, events=events)
        # normalize features
        epochs = self.model.normalize(epochs, mode='test')
        # making decision
        scores = self.model.decision_function(epochs)
        result_index, p = self.decision_logic(scores, events=events)

        print(result_index, p)

        # stop conditions
        if self.trial_cnt >= cfg.exp_config.n_up or p > cfg.exp_config.smart_stopping:
            # trial count reaching the upper limit
            self._q_result.put(result_index)
            # write result index to another file
            self._write_result_ind(result_index)
            # add to buffer
            self.trial_cnt_buf.append(self.trial_cnt)
            self.result_buffer.append(chr(word[result_index]))

            # reset parameters
            self.trial_cnt = 0
            self.char_cnt += 1
            self.score_buffer[:] = 0
        else:
            # confidence is low
            self._q_result.put(-1)
        return

    def decision_logic(self, scores, events=None):
        """

        :param scores: 1d or 2d array of scores of a single trial
        :param events: 2d array 1*total_evt
        """
        self.score_buffer += scores
        if cfg.exp_config.bidir:
            assert events is not None
            # sort events
            events = np.sort(events.squeeze())  # 1d total_evt // 2

            row_score = self.score_buffer[:self.total_evt // 4, 1:].flatten()  # 1d typically 3 * 2 -> 6
            col_score = self.score_buffer[self.total_evt // 4:, 1:].flatten()

            row_score, col_score = list(map(utils.softmax, [row_score, col_score]))

            row_ind = np.argmax(row_score)
            col_ind = np.argmax(col_score)
            p_row = row_score[row_ind]
            p_col = col_score[col_ind]
            prob = p_row * p_col

            n_cls = scores.shape[-1] - 1
            evt_row = events[row_ind // n_cls]
            evt_col = events[col_ind // n_cls + self.total_evt // 4] - (self.total_evt // 2)
            cls_row = row_ind % n_cls + 1  # +1 to match left=1 right=2
            cls_col = col_ind % n_cls + 1

            row = evt_row if cls_row == 1 else (evt_row + self.total_evt // 4) % (self.total_evt // 2)
            col = evt_col if cls_col == 1 else (evt_col + self.total_evt // 4) % (self.total_evt // 2)
        else:
            prob_row = utils.softmax(self.score_buffer[:self.total_evt // 2])
            prob_col = utils.softmax(self.score_buffer[self.total_evt // 2:])
            row = np.argmax(prob_row)
            col = np.argmax(prob_col)
            prob = prob_row[row] * prob_col[col]
        result_index = int(self.total_evt * row // 2 + col)
        return result_index, prob

    def _write_result_ind(self, result):
        result_char = utils.index2char(result)
        with open(self._log_path, 'a') as f:
            f.write('%s %s %d\n' % (self.stim_string[self.char_cnt], result_char, self.trial_cnt))

    def itr(self):
        p = np.mean(list(map(lambda a, b: float(a == b), self.stim_string, self.result_buffer)))
        t_single = 1.2 if cfg.exp_config.bidir else 2.4
        t = np.mean(self.trial_cnt_buf) * t_single
        n = len(word)
        itr = utils.itr(p, n, t)
        self.write_log({'p': p, 'average time': t, 'itr': itr})
        return p, t, itr

    def write_log(self, data_dict):
        with open(self._log_path, 'a') as f:
            for i in data_dict:
                f.write('%s: %s\n' % (i, str(data_dict[i])))
