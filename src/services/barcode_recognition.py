
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yolov5

from src.services.response import Response, Barcode
from src.services.utils import preprocess_image, get_code


class BarcodeRecognition(torch.nn.Module):
    """A class to represent a barcode recognition model.

    Extend methods of basic torch model.

    Attributes:
        _detection_model_path: str - path to detection scripted model.
        _ocr_model_path: str - path to ocr scripted model.
        _device: torch.device - device for model.
        _detection_model: float - model used for detection.
        _ocr_model: float - model used for ocr.
        _detection_size: int - detection image size.
        _conf_thr: float - detection threshold

    Methods:
        predict - predicts probs on raw image.
        predict_proba - predicts classes on raw image.
        _postprocess - predicts probs on tensor.
    """

    def __init__(self, config: Dict):
        """Construct all the necessary attributes for the model object.

        Args:
            config: str - configuration file for model.
        """
        super(BarcodeRecognition, self).__init__()

        self._detection_model_path = config.detection.model_path
        self._ocr_model_path = config.ocr.model_path
        self._device = config.device

        self._detection_model: torch.nn.Module = yolov5.load(
            self._detection_model_path,
            device=self._device,
        )
        self._detection_size: int = config.detection.img_size
        self._conf_thr: float = config.detection.conf_threshold
        self._detection_model.conf = self._conf_thr
        self._detection_model.eval()

        self._ocr_model: torch.nn.Module = torch.jit.load(
            self._ocr_model_path,
            map_location=self._device,
        )
        self._ocr_model.eval()

        self._index2char = self.char2index = dict(
            (self._ocr_model.vocab.index(char), char) for char in self._ocr_model.vocab
        )

    @torch.no_grad()
    def detect_objects(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray], Tuple[float], Tuple[int]]:
        """Predict bbox from image.

        Args:
            image: np.ndarray - RGB image.

        Returns:
            boxes: np.ndarray - object coordinates [x1, y1, x2, y2].
            scores: float - object scores.
            classes: int - object class ids.
        """
        results = self._detection_model(image, size=self._detection_size)

        predictions = results.pred[0]
        boxes = predictions[:, :4].numpy().astype(int)
        scores = predictions[:, 4].numpy().astype(float)
        classes = predictions[:, 5].numpy().astype(int)

        return boxes, scores, classes

    @torch.no_grad()
    def predict_ocr(self, image: np.ndarray) -> np.ndarray:
        """Predict probabilities ocr.

        Args:
            image: np.ndarray - RGB image.

        Returns:
            prediction: np.ndarray - list of probabilities by crnn.
        """
        batch = preprocess_image(image, self._ocr_model.size).to(self._device)
        prediction = self._ocr_model(batch)
        return prediction

    def recognize_barcodes(
        self,
        img: np.ndarray,
    ) -> Dict:
        """Crop bboxes from image.

        Args:
            img: np.ndarray - src RGB img.

        Returns:
            result_barcodes: Dict - result dictionary in api.
        """
        bboxes, _, _ = self.detect_objects(img)
        crops = self._crop_bboxes(img, bboxes)
        barcodes = [self._predict_numbers(crop) for crop in crops]

        result_barcodes = Response()
        for bbox, barcode in zip(bboxes, barcodes):
            barcode = Barcode(bbox={'x_min': bbox[0], 'x_max': bbox[2], 'y_min': bbox[1], 'y_max': bbox[3]},
                              value=barcode)
            result_barcodes.barcodes.append(barcode)
        return result_barcodes

    def _postprocess_ocr(self, prediction: np.ndarray) -> str:
        """Postprocess to get classes from probs.

        Args:
            prediction: np.ndarray - probability predictions by crnn.

        Returns:
            predicted_barcode: str - predicted_barcode.
        """
        prediction = list(get_code(prediction))
        if not len(prediction):
            return ''
        return ''.join([self._index2char[i] for i in prediction])

    def _predict_numbers(self, image: np.ndarray) -> str:
        """Predict numbers on image.

        Args:
            image: np.ndarray - RGB image.

        Returns:
            prediction: str - barcode.
        """
        return self._postprocess_ocr(self.predict_ocr(image))

    def _crop_bboxes(
        self,
        img: np.ndarray,
        bboxes: Tuple[int, int, int, int],
    ) -> List[np.ndarray]:
        """Crop bboxes from image.

        Args:
            img: np.ndarray - src RGB img.
            bboxes: Tuple[int, int, int, int] - barcode bboxes.

        Returns:
            crops: List[np.ndarray] - list of cropped bboxes.
        """
        crops = []
        for bbox in bboxes:
            ymin, ymax = bbox[1], bbox[3]
            xmin, xmax = bbox[0], bbox[2]
            crop = img[ymin:ymax, xmin:xmax]
            h, w = crop.shape[:2]
            if h > w:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            crops.append(crop)
        return crops
