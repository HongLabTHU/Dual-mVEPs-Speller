import glob
import os
import warnings
from datetime import datetime
from copy import deepcopy

import numpy as np
import pyedflib
import scipy.io as sio

from config import cfg
from thirdparty.cerebus import NsxFile, NevFile
from thirdparty.nex import Reader as NexReader
from .utils import find_nearest_time


def _load_neuracle(data_dir):
    """
    neuracle file loader
    :param data_dir: root data dir for the experiment
    :return:
        data: ndarray, (channels, timesteps)
        ch_name: list, name of channels
        timestamp: list, index of trigger
    """
    f = {
        'data': os.path.join(data_dir, 'data.bdf'),
        'evt': os.path.join(data_dir, 'evt.bdf')
    }
    # read data
    f_data = pyedflib.EdfReader(f['data'])
    ch_names = f_data.getSignalLabels()
    data = np.array([f_data.readSignal(i) for i in range(f_data.signals_in_file)])

    # sample frequiencies
    sfreq = f_data.getSampleFrequencies()
    assert np.unique(sfreq).size == 1
    if cfg.amp_info.samplerate != sfreq[0]:
        warnings.warn('Samplerate in config file does not equal to data file record')
    cfg.amp_info.samplerate = int(sfreq[0])

    # read event
    f_evt = pyedflib.EdfReader(f['evt'])
    event, _, _ = f_evt.readAnnotations()
    event = list(map(lambda x: int(x * cfg.amp_info.samplerate), event))

    return data, ch_names, event


def _load_usbamp(data_dir):
    """
        USBAmp file loader
        :param data_dir: root dir
        :return:
            data: ndarray, (channels, timesteps)
            ch_name: list, name of channels
            timestamp: list, index of trigger
        """
    # edf USBAmp
    files = glob.glob(os.path.join(data_dir, '*.edf'))
    assert len(files) == 1
    f = pyedflib.EdfReader(files[0])
    ch_names = f.getSignalLabels()
    # filter channel
    # find trigger channel
    triggers = []
    sig = []
    for i, chan in enumerate(ch_names):
        if 'trigger' in chan:
            triggers.append(i)
        else:
            sig.append(i)
    sigbuf = np.array([f.readSignal(i) for i in range(len(ch_names))])
    ch_names = [ch_names[i] for i in sig]
    trigger = -1
    for ch_ind in triggers:
        if not np.allclose(np.diff(sigbuf[ch_ind]), 0):
            trigger = ch_ind
            break
    diff = np.diff(sigbuf[trigger])
    timestamp = np.nonzero(np.logical_and(diff <= 1, diff >= 0.2))[0].tolist()
    data = sigbuf[sig]
    return data, ch_names, timestamp


def _load_nex(data_dir):
    """
    nex file loader
    :param data_dir:
    :return:
        data: ndarray, shape (ch, timesteps)
        ch_names: list,  name of each channel
        timestamps: list, stimulation onset
    """
    files = glob.glob(os.path.join(data_dir, '*.nex'))
    assert len(files) == 1

    reader = NexReader(useNumpy=True)
    data = reader.ReadNexFile(files[0])

    var = data['Variables']
    ch_names = []
    trigger_ch = None
    con_data = []
    samplerate = cfg.amp_info.samplerate
    for i, ch in enumerate(var):
        if 'CH' in ch['Header']['Name']:
            ch_names.append(ch['Header']['Name'])
            con_data.append(ch['ContinuousValues'])
            samplerate = ch['Header']['SamplingRate']
        if 'digin' == ch['Header']['Name']:
            trigger_ch = i
    if samplerate != cfg.amp_info.samplerate:
        warnings.warn('Samplerate in config file does not equal to data file record, recorded value is %d' % samplerate)
    assert trigger_ch is not None
    timestamp = np.round(data['Variables'][trigger_ch]['Timestamps'] * samplerate).astype(np.int32).tolist()
    con_data = np.array(con_data)
    return con_data, ch_names, timestamp


def _load_cerebus(data_dir):
    # search data_dir
    nsx_files = glob.glob(os.path.join(data_dir, '*.ns*'))
    nev_files = glob.glob(os.path.join(data_dir, '*.nev'))
    assert len(nsx_files) == len(nev_files) == 1
    # loading
    f_data = NsxFile(nsx_files[0])
    f_evt = NevFile(nev_files[0])
    data = f_data.getdata()
    evt = f_evt.getdata()

    f_data.close()
    f_evt.close()

    # some basic information
    samplerate = data['samp_per_s']
    if cfg.amp_info.samplerate != samplerate:
        warnings.warn('Samplerate in config file does not equal to data file record')
    cfg.amp_info.samplerate = samplerate

    timestampresolution = f_evt.basic_header['TimeStampResolution']
    ch_names = []
    for info in f_data.extended_headers:
        ch_names.append(info['ElectrodeLabel'])

    event = evt['dig_events']['TimeStamps'][0]
    event = list(map(lambda x: int(x / timestampresolution * cfg.amp_info.samplerate), event))
    return data['data'], ch_names, event


class Dataset:
    """
    for loading data and event order.
    """
    data_format = {
        'nex': _load_nex,
        'ns3': _load_cerebus,
        'nev': _load_cerebus,
        'edf': _load_usbamp,
        'bdf': _load_neuracle
    }

    def __init__(self, subject, date=None, loaddata=True):
        self.subject = subject
        self._subj_path = os.path.dirname(__file__) + '/../data/' + subject
        if date is None:
            self._date = find_nearest_time(self._subj_path)
        else:
            if isinstance(date, datetime):
                # convert datetime to str
                self._date = date.strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self._date = date
        print(self._date)
        self.root_dir = os.path.join(self._subj_path, self._date)

        # self.montage = OrderedSet(cfg.subj_info.montage)
        self.montage = deepcopy(cfg.subj_info.montage)

        # load stim order
        self.events = self.load_event()

        if loaddata:
            self.load_all()
        else:
            self.data, self.ch_names, self.timestamp, self.montage_indices, self.events_backup = [None] * 5

    def load_all(self):
        # load data and timestamps
        dataarray, ch_names, timestamp = self._load_data()
        timestamp = Dataset.ts_check(timestamp)
        self.data = dataarray
        # list to set
        self.ch_names = ch_names
        self.timestamp = timestamp
        self.montage_indices = self.get_channel_indices(self.montage, self.ch_names)

        self.events_backup = self.events.copy()
        if cfg.exp_config.bidir:
            assert 2 * len(timestamp) == self.events.size, print('Dual-directional: ', len(timestamp), self.events.size)
            self.events = self.events[:, ::2]
        else:
            assert len(timestamp) == self.events.size, print('Unidirectional: ', len(timestamp), self.events.size)

    def _load_data(self):
        """
        Read data according to file format
        :return:
            dataext: str, data file name

        """
        walk_path = self.root_dir
        loader = None
        for f in os.listdir(walk_path):
            _ext = f.split('.')[-1]
            try:
                loader = Dataset.data_format[_ext]
                break
            except KeyError:
                pass
        if loader is None:
            raise FileNotFoundError('No matching data format found')
        return loader(walk_path)

    def load_event(self):
        walk_path = self.root_dir
        file = glob.glob(os.path.join(walk_path, self.subject) + '*')
        assert len(file) == 1
        file = file[0]

        if file.endswith('.mat'):
            raw = sio.loadmat(file)
            order = raw['stim_order']
            order -= 1
            return order.reshape((-1, 12))
        else:
            with open(file) as f:
                stim_order = [[int(x) for x in line.split()] for line in f if len(line) > 1]
            return np.array(stim_order)

    @staticmethod
    def get_channel_indices(target_channels, channels_in_data):
        """
        Get corresponding index number for channels in target channels
        :param target_channels: list, target channel names
        :param channels_in_data: list, all channel names in data source.
        :return:
        """
        indices = []
        # build a dictionary for indexing
        channel_book = {name: i for i, name in enumerate(channels_in_data)}
        for ch in target_channels:
            try:
                indices.append(channel_book[ch])
            except ValueError as err:
                print(err)

        return indices

    @staticmethod
    def ts_check(ts):
        # check time stamp intervals.
        # In our experience, sometimes an accidental wrong trigger may appear at the beginning during recording.
        fs = cfg.amp_info.samplerate
        while len(ts) % 12 and (not (fs * 0.1 <= ts[1] - ts[0] <= fs * 0.3)):
            del ts[0]
        return ts
