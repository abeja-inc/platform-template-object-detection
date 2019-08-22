from typing import Tuple
from utils.data_augumentation import Compose
from utils.data_augumentation import ConvertFromInts
from utils.data_augumentation import ToAbsoluteCoords
from utils.data_augumentation import PhotometricDistort
from utils.data_augumentation import Expand
from utils.data_augumentation import RandomSampleCrop
from utils.data_augumentation import RandomMirror
from utils.data_augumentation import ToPercentCoords
from utils.data_augumentation import Resize
from utils.data_augumentation import SubtractMeans


class DataTransform:
    """preprocess class for image and annotation.
    preprocess is different in training and prediction.

    Attributes:
        - input_size: size of re-sized image
        - color_mean: (B, G, R) mean of each channel
    """

    def __init__(self, input_size: int, color_mean: Tuple[int, int, int]):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                # NOTE: dataset items are not normalized
                ToAbsoluteCoords(),  # de-normalize annotation data
                PhotometricDistort(),  # change image color randomly
                Expand(color_mean),  # expand image canvas
                RandomSampleCrop(),  # extract part of image randomly
                RandomMirror(),
                ToPercentCoords(),  # normalize annotation data with 0-1 range
                Resize(input_size),  # transform image size to input_size × input_size
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),  # transform image size to input_size × input_size
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Args:
            - phase : 'train' or 'val'
        """
        return self.data_transform[phase](img, boxes, labels)
