import torch
from torch.utils.data import DataLoader
from ffllm.data.synthetic_function_calls import FunctionCallDataset
from ffllm.models.ff_transformer import FFTransformer
from ffllm.training.forward_forward import FFConfig, train_epoch_ff
from ffllm.evaluation.metrics import plausibility_accuracy
from torch import optim


def test_train_epoch_ff_runs():
    ds = FunctionCallDataset(num_sequences=20, seq_len=8, vocab_size=32)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    m = FFTransformer(vocab_size=32, d_model=16, n_heads=2, n_layers=1, ffw_hidden=32, max_seq_len=8)
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    cfg = FFConfig(goodness_fn="sumsq", goodness_threshold=2.0, dp_sigma=0.0, clip_norm=1.0)
    stats = train_epoch_ff(m, dl, opt, cfg, torch.device("cpu"))
    assert "loss" in stats


def test_plausibility_metric():
    ds = FunctionCallDataset(num_sequences=10, seq_len=8, vocab_size=32)
    dl = DataLoader(ds, batch_size=5, shuffle=False)
    m = FFTransformer(vocab_size=32, d_model=16, n_heads=2, n_layers=1, ffw_hidden=32, max_seq_len=8)
    acc = plausibility_accuracy(m, dl, torch.device("cpu"))
    assert 0.0 <= acc["plausibility_acc"] <= 1.0
