class Scheduler(object):
    def __init__(self, handlers):
        self.queue = list()
        self.handlers = handlers

    def process(self, queue):
        batches = {handler: list() for handler in self.handlers}

        for task in queue:
            for handler in self.handlers:
                if handler.can_handle(task):
                    break
            else:
                raise RuntimeError(
                    f"The scheduler couldn't find a handler for task {task}")
            batches[handler].append(task)

        new_queue = list()
        for handler, batch in batches.items():
            if not batch:
                continue

            results = handler.handle_batch(batch)
            if isinstance(handler, FinalHandler):
                continue

            new_tasks = list()
            for task, results in zip(batch, results):
                try:
                    new_task = task.gen.send(results)
                except StopIteration:
                    new_task = DoneTask()

                task.metadata.update(new_task.metadata)
                new_task.gen = task.gen
                new_task.metadata = task.metadata
                new_queue.append(new_task)

        return new_queue


class Task(object):
    def __init__(self):
        self.gen = None
        self.metadata = dict()


class Handler(object):
    allowed_task = None

    def handle_batch(self, batch):
        # receives a batch of tasks and handles them
        # can take advantage of batch processing
        raise NotImplementedError()

    def can_handle(self, instance):
        if isinstance(instance, self.allowed_task):
            return True
        else:
            return False


class FinalHandler(Handler):
    pass


class InitTask(Task):
    pass


class DoneTask(Task):
    pass
