import http
import io
import json
import os
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

from dataset import get_dataset_ids
from dataset import set_categories
from data_transform import DataTransform
from parameters import Parameters
from ssd import SSD
from utils.logging import get_logger

logger = get_logger()

# NOTE: define here not to load every time handler function is called
ssd_net: Optional[SSD] = None

# get number of classes of the dataset used in training
job_datasets = json.loads(os.environ.get('TRAINING_JOB_DATASETS'))
train_dataset_id, val_dataset_id = get_dataset_ids(job_datasets)
id2index, index2label = set_categories([train_dataset_id])
num_classes = len(id2index) + 1
print(f'num_classes : {num_classes}')


def initialize_net() -> None:
    global ssd_net

    # if already defined, return it
    if ssd_net is not None:
        print('use cached ssd_net')
        return ssd_net

    # get device ( cpu / gpu ) to be used
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f'device : {device}')

    ssd_cfg = {
        'num_classes': num_classes,  # number of classes including background class
        'input_size': Parameters.IMG_SIZE,
        'bbox_aspect_num': Parameters.BBOX_ASPECT_NUM,
        'feature_maps': Parameters.FEATURE_MAPS,
        'steps': Parameters.STEPS,
        'min_sizes': Parameters.MIN_SIZES,
        'max_sizes': Parameters.MAX_SIZES,
        'aspect_ratios': Parameters.ASPECT_RATIOS,
        'conf_thresh': Parameters.CONF_THRESHOLD,
        'top_k': Parameters.TOP_K,
        'nms_thresh': Parameters.NMS_THRESHOLD
    }
    print(f'initializing ssd with : {ssd_cfg}')
    ssd_net = SSD(phase="inference", cfg=ssd_cfg)

    # load weight created in training
    weight_file_path = os.path.join(Parameters.ABEJA_TRAINING_RESULT_DIR, 'model.pth')
    print(f'weight_file_path : {weight_file_path}')
    # cf. https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
    weight = torch.load(weight_file_path, map_location=device)
    ssd_net.load_state_dict(weight)

    ssd_net = ssd_net.to(device)
    ssd_net.eval()
    return ssd_net


initialize_net()


def handler(request, context):
    logger.info(f'start handling request : {request}, context : {context}')

    contents = request['contents']
    content = contents[0].read()
    f = io.BytesIO(content)
    pil_img = Image.open(f)
    pil_img = np.asarray(pil_img).astype(np.float32)
    height, width, _channels = pil_img.shape

    # TODO: currently image smaller than IMG_SIZE will not be handled properly.
    if height < Parameters.IMG_SIZE or width < Parameters.IMG_SIZE:
        return {
            'status_code': http.HTTPStatus.BAD_REQUEST,
            'content_type': 'application/json',
            'content': {
                'error': http.HTTPStatus.BAD_REQUEST[1],
                'description': f'height and width should be greater than {Parameters.IMG_SIZE}'
            }
        }

    # the dataset's mean rgb values,
    transform = DataTransform(
        input_size=Parameters.IMG_SIZE,
        color_mean=Parameters.MEANS)
    img, _boxes, _labels = transform(pil_img, "val", "", "")
    img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = Variable(img.unsqueeze(0))

    res = ssd_net(x)

    detections = res.data.cpu().detach().numpy()

    find_index = np.where(detections[:, 0:, :, 0] >= Parameters.CONFIDENCE_THRESHOLD)
    detections = res.data.cpu().detach().numpy()

    result = {
        'boxes': [],
        'classes': [],
        'scores': []
    }
    detections = detections[find_index]
    for i in range(len(find_index[1])):
        if (find_index[1][i]) > 0:
            score = float(detections[i][0])
            label_idx = find_index[1][i] - 1
            label = index2label[label_idx]
            bbox = detections[i][1:] * [width, height, width, height]

            result['boxes'].append(bbox)
            result['classes'].append(label)
            result['scores'].append(score)

    logger.info(f'finish handling result : {result}')

    return {
        'status_code': http.HTTPStatus.OK,
        'content_type': 'application/json',
        'content': result
    }
