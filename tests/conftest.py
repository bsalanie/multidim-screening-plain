from pathlib import Path

from dotenv import dotenv_values
from pytest import fixture

from multidim_screening_plain.setup import setup_model


@fixture
def build_model():

    def _build_model(config_file):
        config = dotenv_values(
            Path.cwd() / "multidim_screening_plain" / f"config_{config_file}.env"
        )
        model = setup_model(config)
        return model

    return _build_model
