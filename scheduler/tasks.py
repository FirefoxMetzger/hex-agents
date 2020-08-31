from scheduler.scheduler import Task


class InitExit(Task):
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


class MCTSExpandAndSimulate(Task):
    pass


class UpdateEnv(Task):
    pass


class NNEval(Task):
    pass