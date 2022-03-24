class DummyWriter:
    """
    Class that is used to replace tensorboard SummaryWriter but does not log
    anything. Used to avoid logging anything without tons of if conditions.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self
