import os
from unittest import mock

import numpy as np
import pytest
from pickshot import PickShot, TrainingModel


@pytest.fixture
def mock_model():
    model = mock.Mock()
    model.predict.return_value = np.array([[0.8]])
    return model


def test_pickshot_load():
    pickshot = PickShot.load("saved_models/nuclei_CNN_(50,50).h5")
    assert pickshot.name == "nuclei"
    assert pickshot._image_shape == (50, 50)
    assert hasattr(pickshot.model_storage, "predict")


def test_pickshot_download():
    url = "https://github.com/jkubis96/PickShot/raw/refs/heads/draq5_nuclei_model/saved_models/nuclei_CNN_(50,50).h5"

    pickshot = PickShot.download(url)

    assert pickshot.name == "nuclei"
    assert pickshot._image_shape == (50, 50)
    assert hasattr(pickshot.model_storage, "predict")


def test_predict():
    pickshot = PickShot.load("saved_models/nuclei_CNN_(50,50).h5")

    img_path = os.path.join("data/raw_drqa5/")

    result = pickshot.predict(img_path, ident_part="CH11")

    assert isinstance(result, dict)
    assert "images_ids" in result
    assert "prediciton" in result


def test_trainingmodel_initialization():
    model = TrainingModel()
    assert model.image_shape == (50, 50)
    assert model.epochs_val == 10
    assert model.batch_size_val == 32


def test_trainingmodel_setters():
    model = TrainingModel()
    model.image_shape = (64, 64)
    assert model.image_shape == (64, 64)

    model.test_size_val = 0.3
    assert model.test_size_val == 0.3

    model.epochs_val = 20
    assert model.epochs_val == 20

    model.batch_size_val = 64
    assert model.batch_size_val == 64
