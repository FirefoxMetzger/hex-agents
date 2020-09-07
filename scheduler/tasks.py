from scheduler.scheduler import Task


class InitExit(Task):
    def __init__(self, sample, idx):
        super(InitExit, self).__init__()
        self.metadata = {
            "sample": sample,
            "idx": idx
        }


class NNEval(Task):
    def __init__(self, sim):
        super(NNEval, self).__init__()
        self.sim = sim


class Rollout(Task):
    def __init__(self, sim, action_history):
        super(Rollout, self).__init__()
        self.sim = sim
        self.action_history = action_history
