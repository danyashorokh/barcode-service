import typing as tp

import cv2
import numpy as np
import torch

MAX_UINT8 = 255


def preprocess_image(image: np.ndarray, image_size: tp.Tuple[int, int]) -> torch.Tensor:
    """Препроцессинг имаджнетом.

    :param image: RGB изображение;
    :param image_size: целевой размер изображения;
    :return: батч с одним изображением.
    """
    image = image.astype(np.float32)
    image = cv2.resize(image, image_size) / MAX_UINT8
    image = np.transpose(image, (2, 0, 1))
    image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return torch.from_numpy(image)[None]


def get_code(output: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(output, dim=2).permute(1, 0)
    pred = pred.detach().cpu().numpy()[0]
    pred_code = []
    for i in range(len(pred)):  # noqa: WPS518
        if pred[i] != 0:
            if i == 0 or (pred[i - 1] != pred[i]):
                pred_code.append(pred[i])
    return pred_code
