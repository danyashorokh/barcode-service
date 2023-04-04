import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.barcode_recognition import BarcodeRecognition


@router.post('/recognize')
@inject
def recognize(
    image: bytes = File(),
    service: BarcodeRecognition = Depends(Provide[AppContainer.barcode_recognition]),
):
    """Recognize barcodes on image.

    Args:
        image: bytes - image in byte format.
        service: BarcodeRecognition - di Container entity for barcode recognition.

    Returns:
        output_dict: dict - predict barcodes on image.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.recognize_barcodes(img)


@router.get('/health_check')
def health_check():
    """Check server health.

    Returns:
        res: str - health status.
    """
    return 'OK'
