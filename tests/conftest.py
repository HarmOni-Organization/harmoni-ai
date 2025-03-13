def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "basic: Basic route tests")
    config.addinivalue_line("markers", "recommendations: Movie recommendation tests")
    config.addinivalue_line("markers", "genre: Genre-based recommendation tests")
    config.addinivalue_line("markers", "posters: Movie poster tests")
    config.addinivalue_line("markers", "errors: Error handling tests") 