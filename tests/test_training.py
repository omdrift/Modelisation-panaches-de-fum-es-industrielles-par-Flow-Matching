"""Training recovery, adversarial updates and stable smoke weighting."""
import json
import pytest
import torch


@pytest.mark.parametrize('interrupt', [False, True])
@pytest.mark.parametrize('adversarial', [False, True])
def test_autoencoder_validates_at_its_final_update(tmp_path, monkeypatch, interrupt, adversarial):
    from scientific import runner
    from scientific.protocol import load_protocol
    from lutils.dict_wrapper import DictWrapper
    import model.vqgan.discriminator as discriminator_module
    cfg = load_protocol()
    cfg['evaluation'].update(context_frames=2, horizon=3)
    cfg['training']['max_updates'] = 1
    cfg['training']['discriminator_start'] = 0 if adversarial else 1000
    fail_validation = [interrupt]

    class TinyDataset:
        def __init__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {'frames': torch.ones(36, 3, 8, 8) * 0.2, 'alpha': torch.ones(36, 1, 8, 8)}
    class TinyAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.dropout = torch.nn.Dropout(0.1)
        def forward(self, images):
            return DictWrapper(reconstructed_images=self.dropout(images) * self.weight, vq_loss=self.weight.square())
    class TinyMetric(torch.nn.Module):
        def forward(self, a, b):
            if not torch.is_grad_enabled() and fail_validation[0]:
                fail_validation[0] = False
                raise RuntimeError('final validation interrupted')
            return (a - b).square().mean((1, 2, 3), keepdim=True)
    monkeypatch.setattr(runner, 'WindowDataset', TinyDataset)
    monkeypatch.setattr(runner, 'build_vqvae', lambda *args: TinyAE())
    monkeypatch.setattr(discriminator_module, 'build_discriminator', lambda **kwargs: torch.nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(runner, 'build_lpips', lambda *args: TinyMetric())
    monkeypatch.setattr(runner, 'provenance', lambda *args: {})
    if interrupt:
        with pytest.raises(RuntimeError, match='final validation interrupted'):
            runner.train_autoencoder(cfg, tmp_path, tmp_path / 'run', torch.device('cpu'))
        checkpoint = tmp_path / 'run/checkpoints/step_1.pth'
        assert torch.load(checkpoint, weights_only=True)['extra']['last_validated_step'] == 0
        runner.train_autoencoder(cfg, tmp_path, tmp_path / 'run', torch.device('cpu'), resume=checkpoint)
    else:
        runner.train_autoencoder(cfg, tmp_path, tmp_path / 'run', torch.device('cpu'))
    # Logging may use random numbers internally without perturbing the model.
    class RandomMonitor:
        def __init__(self):
            self.images = []
        def should_log(self, step):
            return True
        def should_media(self, step, final=False):
            return True
        def log(self, step, values, images=None):
            from scientific.monitoring import preserve_rng
            with preserve_rng():
                torch.rand(5)
            if images:
                self.images.append(images)
    monitor = RandomMonitor()
    runner.train_autoencoder(cfg, tmp_path, tmp_path / 'logged', torch.device('cpu'), monitor=monitor)
    original = torch.load(tmp_path / 'run/checkpoints/step_1.pth', weights_only=True)
    logged = torch.load(tmp_path / 'logged/checkpoints/step_1.pth', weights_only=True)
    for key in original['model']:
        torch.testing.assert_close(original['model'][key], logged['model'][key], rtol=0, atol=0)
    for key in original['extra']['discriminator']:
        torch.testing.assert_close(original['extra']['discriminator'][key], logged['extra']['discriminator'][key], rtol=0, atol=0)
    torch.testing.assert_close(original['rng']['torch'], logged['rng']['torch'], rtol=0, atol=0)
    assert monitor.images
    assert (tmp_path / 'logged/metrics.jsonl').exists()
    metrics = json.loads((tmp_path / 'logged/metrics.jsonl').read_text().splitlines()[0])
    assert metrics['train/discriminator_active'] == adversarial
    assert (metrics['train/discriminator'] > 0) == adversarial
    selected = torch.load(tmp_path / 'run/best.pth', weights_only=True)
    assert selected['step'] == 1
    assert selected['extra']['best_validation_lpips'] is not None
    assert selected['extra']['completion']['status'] == 'complete'
    assert selected['extra']['last_validated_step'] == 1



def test_alpha_weighting_is_independent_of_micro_batch_size():
    from scientific.runner import loss_for
    from scientific.protocol import load_protocol
    from lutils.dict_wrapper import DictWrapper
    class FakeModel:
        def __call__(self, frames, alpha_masks):
            return DictWrapper(reconstructed_vectors=frames, target_vectors=torch.zeros_like(frames), smoke_mask=alpha_masks, latent_mask_fraction=torch.tensor(1.))
    frames = torch.tensor([[[[1., 3.]]], [[[2., 4.]]]])
    masks = torch.tensor([[[[1., 0.]]], [[[1., 1.]]]])
    cfg = load_protocol()
    combined = loss_for(FakeModel(), 'flow_alpha', frames, masks, cfg)[0]
    separate = torch.stack([loss_for(FakeModel(), 'flow_alpha', frames[i:i+1], masks[i:i+1], cfg)[0] for i in range(2)]).mean()
    torch.testing.assert_close(combined, separate)

