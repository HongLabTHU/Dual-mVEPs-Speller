import random
from queue import Empty
import warnings

import numpy as np
from psychopy import visual, core

from .Controller import word
from Offline.utils import char2index


class Stimulator:
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255], [160, 82, 45], [100, 120, 100]]
    duration = 150  # in milliseconds
    interval = 200  # in milliseconds
    events_per_trial = 12
    fps = 60
    max_string_display = 20

    def __init__(self,
                 q_result,
                 q_stim,
                 trigger_unit,
                 screen_index=0,
                 stim_string='AHOV29',
                 stim_direction=1,
                 bg_color=(0.4, 0.4, 0.4)):
        """
        Visual stimulation class
        :param q_result: incoming queue to indicate decoded target
        :param q_stim: outcoming queue to transmit stimulation order
        :param trigger_unit: trigger unit for Neuracle or Cerebus amp
        :param screen_index: screen index
        :param stim_string: target string to show up
        :param stim_direction: moving direction of bar: 1 -> left and -1 -> right, None for dual-directional mode
        :param bg_color: background color
        """
        # queue to communicate with processor (protected)
        self._q_result = q_result
        self._q_stim = q_stim

        # set up window
        self.win = visual.Window(units="pix",
                                 fullscr=True,
                                 color=bg_color,
                                 screen=screen_index,
                                 allowGUI=False,
                                 winType='pyglet',
                                 gammaErrorPolicy='warn')
        # trigger unit
        self._trigger = trigger_unit
        # config trigger unit
        self._trigger.config(self.win)

        self._stim_direction = stim_direction
        self.stim_string = stim_string.upper()
        self._frame_max = int(self.duration * self.fps / 1000)
        self._frame_idle = int((self.interval - self.duration) * self.fps / 1000)

        # variables of screen
        self._rects = []
        self._texts = []
        self._bars = []
        # refresh list
        self._refresh_list = []
        self._config_screen()

        # init counting parameters
        self.char_cnt = 0

    def flip(self):
        self._background.draw()
        self._target_string.draw()
        self._result_string.draw()
        for item in self._refresh_list:
            item.draw()
        self.win.flip()

    def wait_start_sig(self):
        def nonblocking_get_start_sig():
            try:
                return self._q_result.get(block=False) != -2
            except Empty:
                return True

        # -1 means continue
        # -2 means start
        while nonblocking_get_start_sig():
            core.wait(0.5, hogCPUperiod=0)
            continue

    def start_logging(self):
        self.win.recordFrameIntervals = True

    def log_info(self):
        intervalsMS = np.array(self.win.frameIntervals) * 1000
        m = np.mean(intervalsMS)
        sd = np.std(intervalsMS)
        msg = "Mean=%.1fms, s.d.=%.2f ms, 99%%CI(frame)=%.2f-%.2f ms"
        distString = msg % (m, sd, m - 2.58 * sd, m + 2.58 * sd)
        nTotal = len(intervalsMS)
        nDropped = sum(intervalsMS > (1500 / self.fps))
        msg = "Dropped/Frames = %i/%i = %.3f%%"
        droppedString = msg % (nDropped, nTotal, 100 * nDropped / float(nTotal))
        print(distString)
        if nDropped > 0:
            warnings.warn(droppedString)

    def show_epoch(self, epoch_ord):
        # add bars to refresh queue
        for bar in self._bars:
            self._refresh_list.append(bar)
        # draw stimulation
        for frame in range(self._frame_max):
            self.draw_bars(epoch_ord=epoch_ord, frame_cnt=frame)

            if frame == 0 and not self._trigger.after_flip():
                self._trigger.send_trigger(0xFF)
            self.flip()
            if frame == 0 and self._trigger.after_flip():
                self._trigger.send_trigger(0xFF)

        # set still for 50 ms
        for frame in range(self._frame_idle):
            self.flip()
            if frame == 0:
                # reset trigger
                self._trigger.reset_trigger()
        # remove bars from refresh list
        for i in range(len(self._bars)):
            self._refresh_list.pop()

    def draw_full_trial(self):
        # In bidir mode, [0, 1, 2, 6, 7, 8] means leftward, and [3, 4, 5, 9, 10, 11] means rightward.
        if self._stim_direction is not None:
            rows = list(range(self.events_per_trial // 2))
            cols = list(range(self.events_per_trial // 2, self.events_per_trial))
        else:
            rows = [(i, i + self.events_per_trial // 4) for i in
                    range(self.events_per_trial // 4)]  # [(0, 3), (1, 4), (2, 5)]
            cols = [(i, i + self.events_per_trial // 4) for i in
                    range(self.events_per_trial // 2, self.events_per_trial * 3 // 4)]  # [(6, 9), (7, 10), (8, 11)]
        random.shuffle(rows)
        random.shuffle(cols)
        stim_order = rows + cols
        for epoch in stim_order:
            self.show_epoch(epoch)
        # wait 0.64s to send stim order, match with offline config + 1 * extra package sending duration (40ms)
        self.wait_with_flip(0.64)
        self._q_stim.put(stim_order)

    def wait_result(self):
        return self._q_result.get()

    def process_result(self, result_index):
        # add new text to self._result_string
        self._result_string.text += chr(word[result_index])
        # fill corresponding rect with black
        self._rects[result_index].setFillColor(color='black')
        self._refresh_list.append(self._rects[result_index])
        # last approximately 500ms
        self.wait_with_flip(0.5)
        self._refresh_list.pop()
        # reset
        self._rects[result_index].setFillColor(color='white')

        # update char count
        self.char_cnt += 1

    def prepare_next_target(self):
        # prepare next char and update strings
        if self._string_process():
            # wait a moment for subject to prepare
            self.wait_with_flip(0.1)
            # draw next target
            target_chr = self.stim_string[self.char_cnt]
            target_index = char2index(target_chr)
            target_rect = self._rects[target_index]
            target_rect.setLineColor(color='red')
            orig_width = target_rect.lineWidth
            target_rect.setLineWidth(5)
            self._refresh_list.append(target_rect)
            # set text to red
            target_text = self._texts[target_index]
            target_text.setColor('red')
            self._refresh_list.append(target_text)
            self.wait_with_flip(0.5)
            self._refresh_list.pop()  # pop text
            self._refresh_list.pop()  # pop rect
            # reset
            target_rect.setLineColor(color='black')
            target_rect.setLineWidth(orig_width)
            target_text.setColor('black')

    def draw_bars(self, epoch_ord, frame_cnt):
        # handle different modes, set the positions of the moving bars
        # draw bars
        if frame_cnt == 0:
            self._setbarColor(shuffle=True)
        if isinstance(epoch_ord, tuple):
            # bars going leftwards
            self._setbarPos(epoch_ord[0], frame_cnt, self._bars[:self.events_per_trial // 2], direction=1)
            # bars going rightwards
            self._setbarPos(epoch_ord[1], frame_cnt, self._bars[self.events_per_trial // 2:], direction=-1)
        else:
            self._setbarPos(epoch_ord, frame_cnt, self._bars)

    def wait_with_flip(self, sec):
        n_frames = int(sec * self.fps)
        for _ in range(n_frames):
            self.flip()

    def _string_process(self):
        """
        skip space or other unrecognized characters
        :return:
            True for char_cnt index still in bound
            False for no more target.
        """
        char_cnt_back = self.char_cnt
        while self.char_cnt < len(self.stim_string) and (ord(self.stim_string[self.char_cnt]) not in word):
            self.char_cnt += 1
        if self.char_cnt >= len(self.stim_string):
            return False
        # refresh display if needed
        if len(self._result_string.text) + self.char_cnt - char_cnt_back >= len(self._target_string.text):
            # buildup new str for display
            display_str = self._split_string(self.char_cnt)
            # refresh stim string & clear result string
            self._target_string.text = display_str
            self._result_string.text = ''
        else:
            # add skipped chars to result string
            # self._result_string.text += self.stim_string[char_cnt_back:self.char_cnt]
            self._result_string.text += ' ' * (self.char_cnt - char_cnt_back)
        return True

    def _config_screen(self):
        width, height = self.win.size
        self._L_squS = round(min(width, height) * 0.9 / 18)
        # define 
        mesh_col, mesh_row = np.meshgrid(np.arange(6), np.arange(6))
        p_col = (-7.5 + 3 * mesh_col) * self._L_squS
        p_row = (6.5 - 3 * mesh_row) * self._L_squS
        self.rect_pos = pos = np.stack((p_col, p_row), axis=2)

        for i in range(pos.shape[0]):
            # i means row
            for j in range(pos.shape[1]):
                # j means col
                rect = visual.Rect(self.win,
                                   width=self._L_squS,
                                   height=self._L_squS,
                                   lineColor='black',
                                   fillColor='white',
                                   pos=(pos[i, j, 0], pos[i, j, 1]),
                                   units='pix')
                self._rects.append(rect)
                text = visual.TextStim(self.win,
                                       text=chr(word[6 * i + j]),
                                       pos=(pos[i, j, 0], pos[i, j, 1]),
                                       color='black',
                                       height=self._L_squS / 3 * 2,
                                       units='pix')
                self._texts.append(text)

        # target string
        # left for horizontal alignment
        display_str = self._split_string(0, self.max_string_display)
        self._target_string = visual.TextStim(self.win,
                                              text=display_str,
                                              font='Courier New',
                                              color='black',
                                              height=50,
                                              pos=(self.rect_pos[0, 0, 0], 0.47 * height),
                                              wrapWidth=self.rect_pos[0, -1, 0] - self.rect_pos[0, 0, 0],
                                              alignHoriz='left')
        self._result_string = visual.TextStim(self.win,
                                              text='',
                                              font='Courier New',
                                              color='red',
                                              height=50,
                                              pos=(self.rect_pos[0, 0, 0], 0.42 * height),
                                              wrapWidth=self.rect_pos[0, -1, 0] - self.rect_pos[0, 0, 0],
                                              alignHoriz='left',
                                              italic=True)
        # left horizontal alignment
        # Update its text when getting new result

        # 6 bars or 12
        bar_num = self.events_per_trial if self._stim_direction is None else self.events_per_trial // 2
        for i in range(bar_num):
            self._bars.append(visual.Line(self.win,
                                          start=(0, 0),
                                          end=(0, -self._L_squS),
                                          lineColor=self.colors[i % len(self.colors)],
                                          lineColorSpace='rgb255',
                                          lineWidth=5))

        self._background = visual.BufferImageStim(self.win, stim=self._rects + self._texts)

    def _split_string(self, start=0, end=None):
        """
        Split based on words, if had
        :return:
        """
        if end is None:
            end = start + self.max_string_display
        if ' ' in self.stim_string[start:end] and end < len(self.stim_string):
            end_ind = self.stim_string.rfind(' ', start, end)
            return self.stim_string[start:end_ind]
        else:
            return self.stim_string[start:end]

    def _setbarPos(self, epoch_cnt, frame_cnt, bars, direction=None):
        """
        Called every frame during stimulation
        :param epoch_cnt:
        :param frame_cnt:
        :param bars:
        :param direction: bar moving direction. 1 -> left -1 -> right
        :return:
        """
        if direction is None and self._stim_direction is not None:
            direction = self._stim_direction
        elif direction is None:
            raise ValueError('No direction specified!')
        step = 2 * self._L_squS / 3 / self._frame_max
        if epoch_cnt >= 6:
            col = self.rect_pos[:, epoch_cnt - 6, 0] + (
                    self._L_squS / 2 - step * (frame_cnt + 1)) * direction
            row = self.rect_pos[:, epoch_cnt - 6, 1] + self._L_squS / 2
        else:
            col = self.rect_pos[epoch_cnt, :, 0] + (self._L_squS / 2 - step * (frame_cnt + 1)) * direction
            row = self.rect_pos[epoch_cnt, :, 1] + self._L_squS / 2

        # set line pos
        for bar, c, r in zip(bars, col, row):
            bar.setPos((c, r))

    def _setbarColor(self, shuffle=True):
        """
        Called every epoch
        :param shuffle:
        :return:
        """
        # coloring rounds
        n_colors = len(self.colors)
        n_rounds = len(self._bars) // n_colors
        for i in range(n_rounds):
            if shuffle:
                random.shuffle(self.colors)
            # set line color
            for j in range(n_colors):
                self._bars[n_colors * i + j].setLineColor(self.colors[j])


def run_exp(q_stim, q_result, screen_index, stim_string, amp, trigger_type, stim_dir):
    if (stim_dir is not None) and (stim_dir not in {-1, 1}):
        raise ValueError("Direction type not supported.")
    if amp.lower() == 'neuracle':
        from Online import Neuracle
        if trigger_type == 'light':
            trigger = Neuracle.TriggerNeuracle(useLightSensor=True, screen_index=screen_index)
        else:
            trigger = Neuracle.TriggerNeuracle(useLightSensor=False)
    elif amp.lower() == 'cerebus':
        from Online import Cerebus

        trigger = Cerebus.TriggerCerebus()
    elif amp.lower() == 'debug':
        from Online.AmpInterface import TriggerUnit

        trigger = TriggerUnit()
    else:
        raise ValueError('Undefined amplifier')

    stimulator = Stimulator(q_result=q_result,
                            q_stim=q_stim,
                            trigger_unit=trigger,
                            screen_index=screen_index,
                            stim_string=stim_string,
                            stim_direction=stim_dir)
    stimulator.flip()
    # block process and wait for starting signal
    stimulator.wait_start_sig()
    stimulator.start_logging()
    while stimulator.char_cnt < len(stimulator.stim_string):
        stimulator.draw_full_trial()
        result = stimulator.wait_result()
        if result >= 0:
            stimulator.process_result(result_index=result)
            stimulator.flip()
            stimulator.wait_with_flip(0.3)
            stimulator.prepare_next_target()
        stimulator.wait_with_flip(0.5)
    stimulator.flip()
    stimulator.log_info()
