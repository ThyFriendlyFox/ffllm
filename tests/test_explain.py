from ffllm.assistant.explain import explain_training_findings


def test_explain_fallback_runs():
    # No API keys in CI; should fallback to built-in summary
    out = explain_training_findings('configs/ci.yaml', provider='openai', model='gpt-4o-mini')
    assert isinstance(out, str) and len(out) > 0
