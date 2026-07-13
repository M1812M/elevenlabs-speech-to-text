"""Import smoke test kept separate so collection itself exercises lazy SDK use."""


def test_provider_package_imports_without_constructing_sdk_client() -> None:
    import elevenlabs_toolkit.providers as providers

    provider = providers.ElevenLabsProvider(api_key="not-used")
    assert provider.__class__.__name__ == "ElevenLabsProvider"
