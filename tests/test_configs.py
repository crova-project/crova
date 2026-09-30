from pathlib import Path

import pytest

from crova import io
from crova.cli import _override, main

CONFIGS = sorted(Path(__file__).parents[1].glob("configs/**/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(p.parents[1])))
def test_config_parses(path):
    config = io.load_config(path)
    assert config and all(isinstance(k, str) for k in config)


def test_override_uses_yaml_values():
    config = _override({"a": 1}, ["a=2", "splits=[train]", "reference=null", "name=x"])
    assert config == {"a": 2, "splits": ["train"], "reference": None, "name": "x"}


def test_cli_help():
    with pytest.raises(SystemExit):
        main(["--help"])
