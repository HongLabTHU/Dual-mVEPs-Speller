"""
This test script sends trigger every 200ms with 6 of them in one group. The interval between groups are 1.14 s
and the total number of groups is 50.
"""

from Online.Neuracle import TriggerNeuracle, Neuracle
import time

n_channel = 8

if __name__ == '__main__':
    data_client = Neuracle(n_channel=n_channel + 1, samplerate=1000)
    trigger = TriggerNeuracle()
    for i in range(50):
        for j in range(6):
            tic = time.time()
            trigger.send_trigger(0xff)
            toc = time.time()
            time.sleep(0.2 - toc + tic)
        # wait for 0.64 second
        time.sleep(0.64)
        timestamps, _ = data_client.get_trial_data()
        print(timestamps)
        # write received timestamps to log
        with open('trigger.log', 'a') as f_log:
            for t in timestamps:
                f_log.write('%d ' % t)
            f_log.write('\n')
        time.sleep(0.5)
    data_client.close()
