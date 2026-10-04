import torch
from torch.optim import SGD

from training.training_loop import _load_checkpoint


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

    step = _load_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert step == 12
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
