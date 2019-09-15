class TriggerUnit:
    def config(self, *args, **kwargs):
        pass

    def send_trigger(self, data):
        return True

    def reset_trigger(self):
        return

    def after_flip(self):
        return True


class AmpDataClient:
    def get_trial_data(self):
        return

    def close(self):
        return
