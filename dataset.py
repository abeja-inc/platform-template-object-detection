import io
from typing import Dict, List, Tuple, Optional

from abeja.datasets import Client
from abeja.datasets.dataset_item import DatasetItem
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data


def get_dataset_ids(datasets: dict) -> Tuple[str, Optional[str]]:
    if len(datasets) == 1:
        train_dataset_id = list(datasets.values())[0]
        val_dataset_id = None
    elif len(datasets) == 2:
        assert 'val' in datasets, "alias named `val` is required for either of the two dataset_ids."
        val_dataset_id = datasets.pop('val')
        train_dataset_id = list(datasets.values())[0]
    else:
        raise NotImplementedError('more than two dataset not supported yet.')
    return train_dataset_id, val_dataset_id


def set_categories(dataset_ids: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Set categories from Datasets.
    :param dataset_ids: Dataset IDs. list format.
    :return id2index: Map of label_id to training index.
    :return index2label: Map of training index to label.
    """
    client = Client()
    id2index = dict()
    index2label = dict()
    index = 0

    last_dataset = None
    for dataset_id in dataset_ids:
        dataset = client.get_dataset(dataset_id)
        if len(dataset.props.get('categories', [])) > 1:
            raise NotImplementedError('more than one category not supported yet.')

        # check if all categories are same
        if last_dataset is not None:
            if last_dataset.props['categories'] != dataset.props['categories']:
                raise NotImplementedError('different categories among datasets not supported yet.')
        last_dataset = dataset

    category_0 = last_dataset.props['categories'][0]  # FIXME: Allow category selection
    for label in category_0['labels']:
        label_id = label['label_id']
        label_name = label['label']
        if label_id not in id2index:
            id2index[label_id] = index
            index2label[index] = label_name
            index += 1
    return id2index, index2label


def load_dataset_from_api(dataset_id, max_num=None, organization_id=None, credential=None):
    client = Client(organization_id, credential)
    dataset = client.get_dataset(dataset_id)
    
    if max_num is not None:
        dataset_list = dataset.dataset_items.list(prefetch=False)
        ret = []
        for d in dataset_list:
            ret.append(d)
            if len(ret) > max_num:
                break
        return ret
    else:
        return dataset.dataset_items.list(prefetch=True)


class ConcatenatedDataset(data.Dataset):
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, index):
        if index < 0:
            raise IndexError
        for dataset in self._datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError


class ABEJAPlatformDataset(data.Dataset):
    """class that created VOC2012 dataset

    Attributes:
    - dataset_items: list of dataset item
    - phase: 'train' or 'test'
    - transform: instance of preprocess class which is callable
    """
    def __init__(self, dataset_items: List[DatasetItem], phase, transform):
        self.dataset_items = dataset_items
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """number of images"""
        return len(self.dataset_items)

    def __getitem__(self, index):
        """get preprocessed tensor image and annotation"""
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        """fetch preprocessed tensor image, annotation, height and width of image"""
        # 1. read image
        dataset_item = self.dataset_items[index]
        source_data = dataset_item.source_data[0]
        content = source_data.get_content()
        file_like_object = io.BytesIO(content)

        img = Image.open(file_like_object)
        img = np.asarray(img.convert('RGB')).astype(np.float32)
        height, width, channels = img.shape

        # 2. build list of annotation
        anno_list = []
        if not dataset_item.attributes['detection']:
            # NOTE: add annotation as background ( no object )
            # label_id: -1 will be incremented and used for background
            dataset_item.attributes['detection'] = [
                {
                    'label_id': -1,
                    'rect': {
                        'xmin': width,
                        'ymin': height,
                        'xmax': width,
                        'ymax': height,
                    }
                }
            ]
        for detection in dataset_item.attributes['detection']:
            label_id = detection['label_id']
            rect = detection['rect']
            # normalized 0 to 1
            anno_list.append([
                rect['xmin'] / width,
                rect['ymin'] / height,
                rect['xmax'] / width,
                rect['ymax'] / height,
                label_id
            ])

        anno_list = np.array(anno_list)

        # 3. perform preprocess
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        # change channel color order from BGR to RGB
        # and convert order from (h, w, c) to (c, h, w)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # create np.array that sets BBox and label, gt means `ground truth`
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
