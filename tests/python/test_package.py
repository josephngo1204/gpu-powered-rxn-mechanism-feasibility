import saddle


def test_package_exposes_version() -> None:
    assert saddle.__version__ == "0.0.1"
