import socket
import struct
import threading

import numpy as np

from Online.AmpInterface import AmpDataClient


class Neuracle(AmpDataClient):
    UPDATE_INTERVAL = 0.04
    BYTES_PER_NUM = 4
    BUFFER_LEN = 4  # in secondes

    def __init__(self, n_channel=9, samplerate=1000, host='localhost', port=8712):
        self.n_channel = n_channel
        self.chunk_size = int(self.UPDATE_INTERVAL * samplerate * self.BYTES_PER_NUM * n_channel)
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.buffer = []
        self.max_buffer_length = int(self.BUFFER_LEN / self.UPDATE_INTERVAL)
        self._host = host
        self._port = port
        # thread lock
        self.lock = threading.Lock()
        self.__datathread = threading.Thread(target=self.__recv_loop)

        # start client
        self.config()

    def config(self):
        self.__sock.connect((self._host, self._port))
        self.__run_forever()

    def is_active(self):
        return self.__sock.fileno() != -1

    def close(self):
        self.__sock.close()
        self.__datathread.join()

    def __recv_loop(self):
        while self.__sock.fileno() != -1:
            try:
                data = self.__sock.recv(self.chunk_size)
            except OSError:
                break
            if len(data) % 4 != 0:
                continue
            self.lock.acquire()
            self.buffer.append(data)
            # remove old data
            if len(self.buffer) == self.max_buffer_length:
                del self.buffer[0]
            self.lock.release()

    def __run_forever(self):
        self.__datathread.start()

    def get_trial_data(self):
        """
        called to copy trial data from buffer
        :return:
        timestamps: list of timestamp
        data: ndarray with shape of (channels, timesteps)
        """
        self.lock.acquire()
        raw_data = self.buffer.copy()
        self.buffer.clear()
        self.lock.release()
        total_data = b''.join(raw_data)
        byte_data = bytearray(total_data)
        if len(byte_data) % 4 != 0:
            raise ValueError
        data = np.frombuffer(byte_data, dtype='<f')
        data = np.reshape(data, (-1, self.n_channel))
        timestamps = np.nonzero(data[:, -1])[0].tolist()
        return timestamps, data[:, :-1].T
