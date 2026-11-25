from pathlib import Path

from dotenv import dotenv_values
from pytest import fixture

from multidim_screening_plain.setup import setup_model

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "multidim_screening_plain"


@fixture
def build_model():
    """returns a function that builds the `ScreeningModel` for
    `model_type` (e.g. "insurance") and `model_instance` (e.g. "d2_m2")
    from its configuration file"""

    def _build_model(model_type: str, model_instance: str):
        config_file = (
            PACKAGE_DIR
            / model_type
            / model_instance
            / f"config_{model_type}_{model_instance}.env"
        )
        config = dotenv_values(config_file)
        return setup_model(config, model_type, model_instance)

    return _build_model
