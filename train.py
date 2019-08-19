import os
import numpy as np
import time
import random

from tensorboardX import SummaryWriter
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from data_loader import od_collate_fn
from data_transform import DataTransform
from dataset import ABEJAPlatformDataset
from dataset import get_dataset_ids
from dataset import set_categories
from dataset import load_dataset_from_api
from parameters import Parameters
from ssd import SSD
from ssd import MultiBoxLoss
from tools import download
from utils.callbacks import Statistics

if Parameters.RANDOM_SEED is not None:
    torch.manual_seed(Parameters.RANDOM_SEED)
    np.random.seed(Parameters.RANDOM_SEED)
    random.seed(Parameters.RANDOM_SEED)


use_cuda = torch.cuda.is_available()
# NOTE: multi gpu not supported yet
device = torch.device("cuda" if use_cuda else "cpu")
print("device: ", device)

statistics = Statistics(Parameters.EPOCHS)

log_path = os.path.join(Parameters.ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)


def handler(context):
    print(f'start training with parameters : {Parameters.as_dict()}, context : {context}')

    try:
        dataset_alias = context.datasets    # for older version
    except AttributeError:
        dataset_alias = context['datasets']

    train_dataset_id, val_dataset_id = get_dataset_ids(dataset_alias)

    id2index, _ = set_categories(list(dataset_alias.values()))
    num_classes = len(id2index)
    num_classes += 1    # add for background class

    print(f'number of classes : {num_classes}')

    dataset_items = list(load_dataset_from_api(
        train_dataset_id, max_num=Parameters.MAX_ITEMS))

    random.shuffle(dataset_items)
    if val_dataset_id is not None:
        val_dataset_items = list(load_dataset_from_api(
            val_dataset_id, max_num=Parameters.MAX_ITEMS))
        random.shuffle(val_dataset_items)
        train_dataset_items = dataset_items
    else:
        test_size = int(len(dataset_items) * Parameters.TEST_SIZE)
        train_dataset_items, val_dataset_items = dataset_items[test_size:], dataset_items[:test_size]

    train_dataset = ABEJAPlatformDataset(train_dataset_items, phase="train", transform=DataTransform(
        Parameters.IMG_SIZE, Parameters.MEANS))

    val_dataset = ABEJAPlatformDataset(val_dataset_items, phase="val", transform=DataTransform(
        Parameters.IMG_SIZE, Parameters.MEANS))

    print(f'train dataset : {len(train_dataset)}')
    print(f'val dataset : {len(val_dataset)}')

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=Parameters.BATCH_SIZE,
        shuffle=Parameters.SHUFFLE, collate_fn=od_collate_fn)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=Parameters.BATCH_SIZE,
        shuffle=False, collate_fn=od_collate_fn)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    print(f'data loaders : {dataloaders_dict}')

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
    net = SSD(phase="train", cfg=ssd_cfg)

    # TODO: better to host this file by ourselves
    # https://github.com/amdegroot/ssd.pytorch#training-ssd
    url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
    weight_file = os.path.join(Parameters.ABEJA_TRAINING_RESULT_DIR, 'vgg16_reducedfc.pth')
    download(url, weight_file)

    vgg_weights = torch.load(weight_file)
    print('finish loading base network...')
    net.vgg.load_state_dict(vgg_weights)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:  # in case of bias
                nn.init.constant_(m.bias, 0.0)

    # apply initial values of He
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    # configure loss function
    criterion = MultiBoxLoss(
        jaccard_thresh=Parameters.OVERLAP_THRESHOLD,
        neg_pos=Parameters.NEG_POS,
        device=device)

    # configure optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=Parameters.LR,
        momentum=Parameters.MOMENTUM,
        dampening=Parameters.DAMPENING,
        weight_decay=Parameters.WEIGHT_DECAY,
        nesterov=Parameters.NESTEROV)

    # move network to device
    net.to(device)

    # NOTE: This flag allows to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware.
    # cf. https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    torch.backends.cudnn.benchmark = True

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    latest_epoch_train_loss = epoch_train_loss
    latest_epoch_val_loss = epoch_val_loss

    for epoch in range(Parameters.EPOCHS):

        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch + 1, Parameters.EPOCHS))
        print('-------------')

        # loop of train and validation for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('（train）')
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print('-------------')
                    print('（val）')
                else:
                    # perform validation once every ten times
                    continue

            # loop each mini-batch from data loader
            for images, targets in dataloaders_dict[phase]:

                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # initialize optimizer
                optimizer.zero_grad()

                # calculate forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)

                    # calculate loss
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == 'train':
                        # back propagate when training
                        loss.backward()  # calculate gradient

                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=Parameters.CLIP_VALUE)

                        optimizer.step()  # update parameters

                        if iteration % 10 == 0:  # display loss once every ten iterations
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('iter {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()

        # loss and accuracy rate of each phase of epoch
        t_epoch_finish = time.time()

        # keep latest epoch loss
        if epoch_train_loss != 0.0:
            latest_epoch_train_loss = epoch_train_loss
        if epoch_val_loss != 0.0:
            latest_epoch_val_loss = epoch_val_loss

        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Loss:{:.4f}'.format(
            epoch + 1, latest_epoch_train_loss, latest_epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        statistics(epoch + 1, latest_epoch_train_loss, None, latest_epoch_val_loss, None)

        writer.add_scalar('main/loss', latest_epoch_train_loss, epoch + 1)
        if (epoch + 1) % 10 == 0:
            writer.add_scalar('test/loss', latest_epoch_val_loss, epoch + 1)

            model_path = os.path.join(Parameters.ABEJA_TRAINING_RESULT_DIR, f'ssd300_{str(epoch + 1)}.pth')
            torch.save(net.state_dict(), model_path)

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

    torch.save(net.state_dict(), os.path.join(Parameters.ABEJA_TRAINING_RESULT_DIR, 'model.pth'))
