import argparse
import time
from multiprocessing import Queue, Process, freeze_support
import keyboard

import Online.Controller as Controller
import Online.Stimulator as Stimulator
from config import cfg, merge_cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Online typing BCI main entry'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for subject',
        default=None,
        type=str
    )
    parser.add_argument(
        '--screen-index',
        '-s',
        dest='screen_index',
        help='Screen index',
        default=-1,
        type=int
    )

    # testing parameters
    parser.add_argument(
        '--test',
        '-t',
        dest='test',
        help='activate testing mode',
        action='store_true'
    )
    parser.add_argument(
        '--date',
        '-d',
        dest='model_date',
        help='model datetime',
        type=str,
        default=None
    )
    return parser.parse_args()


def quit_process():
    Controller.quit_flag = True


def online_main(screen_index, test, cfg_file=None, model_date=None):
    if cfg_file is not None:
        merge_cfg_from_file(cfg_file)
    # q_result: controller -> screen (sending judgement result and start/quit signal)
    # q_stim: screen -> controller (sending event order)
    q_result = Queue()
    q_stim = Queue()
    # create stimulator object
    stim_string = cfg.exp_config.train_string if not test else cfg.exp_config.test_string
    kwargs = {
        'q_stim': q_stim,
        'q_result': q_result,
        'screen_index': screen_index,
        'stim_string': stim_string,
        'amp': cfg.amp_info.amp,
        'trigger_type': cfg.amp_info.trigger_type,
        'stim_dir': cfg.exp_config.stim_dir if not cfg.exp_config.bidir else None  # None for bidir
    }

    print('Configuration finished. Start process.')
    # start process
    process = Process(target=Stimulator.run_exp, kwargs=kwargs)
    process.start()

    # main process
    if test:
        # testing mode
        # create data_client object
        n_channel = len(cfg.subj_info.montage)
        if cfg.amp_info.amp == 'neuracle':
            from Online import Neuracle

            data_client = Neuracle.Neuracle(n_channel=n_channel + 1,  # +1 for trigger channel
                                            samplerate=cfg.amp_info.samplerate)

        else:
            raise ValueError("Unexpected amplifier type")

        controller = Controller.TestingController(q_stim=q_stim, q_result=q_result, dataclient=data_client,
                                                  model_date=model_date, stim_string=stim_string)
    else:
        # training mode
        controller = Controller.TrainingController(q_stim=q_stim, q_result=q_result, stim_string=stim_string)
    # write exp config info to log file
    controller.write_exp_log()
    # waiting start signal
    keyboard.wait('s')
    # put starting signal into q_result
    q_result.put(-2)
    # set quit hotkey
    keyboard.add_hotkey('q', quit_process)
    while controller.char_cnt < len(controller.stim_string) and not Controller.quit_flag:
        controller.run()
        time.sleep(0.05)
    # close events file io
    controller.close()
    if test:
        # writing down itr
        print('accu: %.2f, average time: %.2f, itr: %.2f' % controller.itr())
        # turn off data client thread
        data_client.close()
    # terminate process
    if Controller.quit_flag:
        process.terminate()
    process.join()


if __name__ == '__main__':
    # freeze support first
    freeze_support()
    args = parse_args()
    online_main(args.screen_index, args.test, args.cfg_file, args.model_date)
