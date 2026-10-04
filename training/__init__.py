"""Training entry points, imported lazily to keep helpers lightweight."""


def __getattr__(name):
    if name == "train":
        from training.training_loop import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
