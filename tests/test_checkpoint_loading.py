import torch
import pytest
from torch.optim import SGD

from training.checkpoints import load_checkpoint


def test_flow_matching_checkpoint_restores_weights_and_step(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "step_12.pth"
    torch.save(
        {
            "step": 12,
            "model": expected,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    step = load_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert step == 12
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_retention_uses_numeric_steps_and_preserves_best(tmp_path):
    from training.checkpoints import save_checkpoint, copy_checkpoint, latest_checkpoint
    model = torch.nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    first = save_checkpoint(model, optimizer, scheduler, tmp_path, 9)
    copy_checkpoint(first, tmp_path / 'best.pth')
    best = (tmp_path / 'best.pth').read_bytes()
    for step in (10, 100):
        save_checkpoint(model, optimizer, scheduler, tmp_path, step)
    assert {p.name for p in (tmp_path / 'checkpoints').iterdir()} == {'step_10.pth', 'step_100.pth'}
    assert latest_checkpoint(tmp_path).name == 'step_100.pth'
    assert (tmp_path / 'best.pth').read_bytes() == best
    assert torch.load(latest_checkpoint(tmp_path), weights_only=True)['step'] == 100


def test_failed_write_keeps_recovery_points_and_cleans_partial_file(tmp_path, monkeypatch):
    from training import checkpoints
    model = torch.nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = [checkpoints.save_checkpoint(model, optimizer, scheduler, tmp_path, step) for step in (1, 2)]
    originals = {p: p.read_bytes() for p in saved}

    def failed_save(state, path):
        path.write_bytes(b'partial')
        raise RuntimeError('PytorchStreamWriter failed writing file')

    monkeypatch.setattr(torch, 'save', failed_save)
    for step in (2, 3):  # A same-step rewrite must also preserve its old bytes.
        with pytest.raises(checkpoints.CheckpointWriteError, match='free disk space'):
            checkpoints.save_checkpoint(model, optimizer, scheduler, tmp_path, step)
    assert {p: p.read_bytes() for p in saved} == originals
    assert {p.name for p in (tmp_path / 'checkpoints').iterdir()} == {'step_1.pth', 'step_2.pth'}


def test_best_copy_failure_keeps_previous_selected_artifact(tmp_path, monkeypatch):
    from training import checkpoints
    source, best = tmp_path / 'latest.pth', tmp_path / 'best.pth'
    source.write_bytes(b'new selected checkpoint')
    best.write_bytes(b'previous best')

    def failed_copy(source, target):
        target.write_bytes(b'partial')
        raise OSError('No space left on device')

    monkeypatch.setattr(checkpoints.shutil, 'copyfile', failed_copy)
    with pytest.raises(checkpoints.CheckpointWriteError):
        checkpoints.copy_checkpoint(source, best)
    assert best.read_bytes() == b'previous best'
    assert set(tmp_path.iterdir()) == {source, best}


@pytest.mark.parametrize('operation', ['save', 'copy'])
def test_tempfile_creation_failure_is_classified_as_storage_error(tmp_path, monkeypatch, operation):
    from training import checkpoints
    path = tmp_path / 'best.pth'
    path.write_bytes(b'previous best')

    def no_space(**kwargs):
        raise OSError('No space left on device')

    monkeypatch.setattr(checkpoints.tempfile, 'NamedTemporaryFile', no_space)
    with pytest.raises(checkpoints.CheckpointWriteError, match='free disk space'):
        if operation == 'save':
            checkpoints.atomic_save({}, path)
        else:
            checkpoints.copy_checkpoint(tmp_path / 'source', path)
    assert path.read_bytes() == b'previous best'
