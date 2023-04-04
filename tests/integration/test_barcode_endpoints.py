from http import HTTPStatus

from fastapi.testclient import TestClient


def test_recognize(client: TestClient, sample_image_bytes: bytes):
    """Test recognize response.

    Args:
        client: TestClient - test client.
        sample_image_bytes: bytes - image.
    """
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/barcodes/recognize', files=files)

    assert response.status_code == HTTPStatus.OK

    predicted_barcodes = response.json()['barcodes']

    assert isinstance(predicted_barcodes, list)
