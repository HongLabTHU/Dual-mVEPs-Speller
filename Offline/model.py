import os
from datetime import datetime

import numpy as np
from mne.preprocessing.xdawn import _XdawnTransformer
from mne.decoding import Vectorizer
from scipy import signal
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from Offline import utils
from config import cfg


class FeatExtractor:
    def __init__(self, sfreq, band_erp=None, n=None):
        """
        Constructor of class featExtractor
        Pay attention: set erp_band to (1, ..) may result in really bad fir filter or extremely unstable iir filter
        So I recommend using lowpass filter to extract erp...
        Run testOfflineUtils to see if your modifications can pass the test.
        :param sfreq: int sample frequency
        :param band_erp: tuple f band of erp
        :param band_hg: tuple f band of hg
        :param method: 'iir' or 'fir'
        :param n: order of fir filter
        """
        assert band_erp is not None
        band_erp = np.array(band_erp, dtype=np.float64)
        self._band_erp = band_erp
        self.sfreq = sfreq

        if n is None:
            n = int(sfreq / 5)

        if band_erp is not None:
            if band_erp.size == 1:
                b_erp = signal.firwin(n + 1, band_erp, fs=sfreq, pass_zero=True)
            else:
                b_erp = signal.firwin(n + 1, band_erp, fs=sfreq, pass_zero=False)
            self.filter_erp = (b_erp, 1)

    def __call__(self, data):
        """
        extract features
        :param data: ndarray with the last axis "timesteps"
        :return:
        """
        erp = signal.filtfilt(*self.filter_erp, data, axis=-1)
        return erp


class Model:
    def __init__(self, subject=None, date=None, mode='train', **kwargs):
        if subject is None:
            subject = cfg.subj_info.subjname
        self._subj_path = os.path.dirname(__file__) + '/../data/' + subject
        if date is None:
            self._date = utils.find_nearest_time(self._subj_path)
        else:
            if isinstance(date, datetime):
                # convert datetime to str
                self._date = date.strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self._date = date

        self.mode = mode.lower()
        assert self.mode in ['train', 'test']
        if self.mode == 'test':
            # loading trained coefficient
            self.data_dict = np.load(os.path.join(self._subj_path, self._date, 'coef.npz'))
            # loading trained model
            self.__cls = joblib.load(os.path.join(self._subj_path, self._date, 'model.pkl'))
            self._ch_ind = self.data_dict['ind_ch_scores']
        else:
            self.data_dict = {}
            C = kwargs.pop('C', 1)
            n_components = kwargs.pop('n_components', 3)
            self.__cls = make_pipeline(
                _XdawnTransformer(n_components=n_components),
                Vectorizer(),
                StandardScaler(),
                LogisticRegression(C=C, class_weight='balanced', solver='liblinear', multi_class='ovr')
            )

    @property
    def ch_ind(self):
        return self._ch_ind

    @ch_ind.setter
    def ch_ind(self, value):
        value.sort()
        self._ch_ind = value
        self.data_dict['ind_ch_scores'] = value

    def fit(self, X, y):
        """
        Fit model with labeled data
        :param X: epoch data (n_epoch, n_chan, n_times)
        :param y: labels (n_epoch, )
        :return:
        """
        self.__cls.fit(X, y)

    def decision_function(self, X):
        """
        Give prediction on incoming epoch data
        :param X: epoch data
        :return: predictions
        """
        y = self.__cls.decision_function(X)
        return y

    @staticmethod
    def raw2epoch(raw_data, timestamps, events):
        """
        Process raw data to form downsampled epoch data
        :param raw_data: raw feature data (n_chan, n_steps)
        :param timestamps: list of timestamps
        :param events: 2d array, (n_trials, n_epochs_per_trial)
        :return: downsampled epochs
        """

        if cfg.subj_info.type == 'eeg':
            # EEG
            t_base = (cfg.off_config.start, cfg.off_config.end, cfg.amp_info.samplerate)
            epochs = utils.cut_epochs(t_base, raw_data, timestamps)
            epochs = utils.sort_epochs(epochs, events)
            # detrend and correct baseline
            epochs = signal.detrend(epochs, axis=-1, type='linear')
            epochs = utils.apply_baseline(t_base, epochs)

            # apply new time window
            epochs = utils.timewindow((t_base[0], t_base[1]), cfg.off_config.time_window, epochs)
        else:
            raise KeyError('Unsupported recording type.')
        # down sampling
        down_ratio = int(cfg.amp_info.samplerate / cfg.off_config.downsamp)
        epochs = epochs[..., ::down_ratio]
        return epochs

    def dump(self):
        # save coef
        if self.data_dict:
            np.savez(os.path.join(self._subj_path, self._date, 'coef.npz'), **self.data_dict)
        # save model
        if self.__cls is not None:
            joblib.dump(self.__cls, os.path.join(self._subj_path, self._date, 'model.pkl'))
