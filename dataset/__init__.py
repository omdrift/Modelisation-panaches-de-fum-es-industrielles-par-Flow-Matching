"""Dataset package with optional HDF5 dependencies loaded on demand."""


def __getattr__(name):
    if name == "VideoDataset":
        from dataset.video_dataset import VideoDataset

        return VideoDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
