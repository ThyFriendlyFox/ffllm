import torch
from ffllm.models.ff_transformer import FFTransformer


def test_model_forward_shapes():
    m = FFTransformer(vocab_size=32, d_model=32, n_heads=2, n_layers=1, ffw_hidden=64, max_seq_len=16)
    toks = torch.randint(0, 32, (2, 8))
    logits, acts = m(toks)
    assert logits.shape == (2, 8, 32)
    assert len(acts) > 0


def test_model_generate():
    m = FFTransformer(vocab_size=16, d_model=16, n_heads=2, n_layers=1, ffw_hidden=32, max_seq_len=8)
    toks = torch.randint(0, 16, (1, 4))
    out = m.generate(toks, max_new_tokens=4)
    assert out.shape[1] == 8
