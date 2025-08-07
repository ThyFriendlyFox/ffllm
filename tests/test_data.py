from ffllm.data.synthetic_function_calls import FunctionCallDataset


def test_dataset_lengths():
    ds = FunctionCallDataset(num_sequences=10, seq_len=8, vocab_size=32)
    assert len(ds) == 10
    p, n = ds[0]
    assert p.shape[0] == 8 and n.shape[0] == 8
