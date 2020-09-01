from scheduler.scheduler import Task


class UpdateMetadata(Task):
    pass


class InitExit(Task):
    def __init__(self, sample, idx):
        super(InitExit, self).__init__()
        self.metadata = {
            "sample": sample,
            "idx": idx
        }


class ExpandAndSimulate(Task):
    def __init__(self, sim, action_history):
        super(ExpandAndSimulate, self).__init__()
        self.sim = sim
        self.metadata = {
            "action_history": action_history
        }


class MCTSExpandAndSimulate(Task):
    def __init__(self, sim, action_history):
        super(MCTSExpandAndSimulate, self).__init__()
        self.sim = sim
        self.action_history = action_history


class NNEval(Task):
    def __init__(self, sim):
        super(NNEval, self).__init__()
        self.sim = sim
