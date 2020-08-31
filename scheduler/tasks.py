from scheduler.scheduler import InitTask, Task


class InitExit(InitTask):
    def __init__(self, sample, idx):
        super(InitExit, self).__init__()
        self.metadata = {
            "sample": sample,
            "idx": idx
        }


class ExpandAndSimulate(Task):
    def __init__(self, action_history):
        super(ExpandAndSimulate, self).__init__()
        self.metadata = {
            "action_history": action_history
        }
