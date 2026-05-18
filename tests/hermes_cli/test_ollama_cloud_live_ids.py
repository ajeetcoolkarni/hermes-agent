"""Regression tests for Ollama Cloud live model ID handling."""


def test_ollama_cloud_model_fetch_preserves_live_ids(monkeypatch, tmp_path):
    """Live Ollama Cloud IDs must be returned exactly as the API reports them.

    ``:cloud``/``-cloud`` suffixes are legacy local-vs-cloud hints from fallback
    catalogs.  Live ``https://ollama.com/v1/models`` already returns the IDs
    the provider accepts, so /model must not rewrite them.
    """
    import hermes_cli.models as models_mod

    monkeypatch.setattr(models_mod, "_ollama_cloud_cache_path", lambda: tmp_path / "ollama.json")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(
        models_mod,
        "fetch_api_models",
        lambda api_key, base_url, timeout=8.0: ["deepseek-v4-pro", "gemma4:31b"],
    )
    monkeypatch.setattr(
        "agent.models_dev.list_agentic_models",
        lambda provider: ["deepseek-v4-pro:cloud", "minimax-m2.7:cloud"],
    )

    ids = models_mod.fetch_ollama_cloud_models(force_refresh=True)

    assert ids == ["deepseek-v4-pro", "gemma4:31b", "minimax-m2.7"]
    assert all(not mid.endswith(":cloud") and not mid.endswith("-cloud") for mid in ids)
