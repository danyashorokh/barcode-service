from copy import deepcopy

import numpy as np

from src.containers.containers import AppContainer


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    """Test all functions in weather classifier.

    Args:
        sample_image_np: np.ndarray - image.
        app_container: AppContainer - di container.
    """
    barcode_recognition = app_container.barcode_recognition()
    barcode_recognition.recognize_barcodes(sample_image_np)
    barcode_recognition.detect_objects(sample_image_np)


def test_bboxes_shape(app_container: AppContainer, sample_image_np: np.ndarray):
    """Test bbox not exceed image shape.

    Args:
        sample_image_np: np.ndarray - image.
        app_container: AppContainer - di container.
    """
    barcode_recognition = app_container.barcode_recognition()
    bboxes, _, classes = barcode_recognition.detect_objects(sample_image_np)
    h, w, c = sample_image_np.shape
    for bbox, class_id in zip(bboxes, classes):
        assert 0 <= bbox[0] <= w
        assert 0 <= bbox[2] <= w
        assert 0 <= bbox[1] <= h
        assert 0 <= bbox[3] <= h
        assert class_id == 0


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    """Test mutation of original image.

    Args:
        sample_image_np: np.ndarray - image.
        app_container: AppContainer - di container.
    """
    initial_image = deepcopy(sample_image_np)
    barcode_recognition = app_container.barcode_recognition()
    barcode_recognition.recognize_barcodes(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)
