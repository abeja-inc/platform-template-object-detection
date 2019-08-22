from math import sqrt
from itertools import product

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

from utils.match import match


def make_vgg():
    """make vgg module composed of 34 layers"""
    layers = []
    in_channels = 3  # number of color channel

    # number of convolution layer and max pooling channels used in vgg.
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceil_mode rounds up output size with calculated result
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def make_extras():
    """make extras module composed of 8 layers"""
    layers = []
    # number of image channel output from vgg module and input for extra
    in_channels = 1024

    # number of channel for convolution layer
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]

    # ReLU will be prepared in forward in SSD model.

    return nn.ModuleList(layers)


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    """make loc_layers that output offset of default box,
    and make conf_layers that output confidence of each class for the default box
    """

    loc_layers = []
    conf_layers = []

    # 22nd layer of VGG, convolution layer against conv4_3
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # convolution layer against the last layer of VGG
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # convolution layer against extra
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
    """layer that normalizes using L2Norm with scale = 20"""
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        """initialize combined parameters"""
        init.constant_(self.weight, self.scale)  # update weight values with scale

    def forward(self, x):

        # calculate sum of squares of channel of 38 x 38 feature for each channel,
        # and calculate square root it, normalize by division.
        # tensor size of norm is torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        # transform into torch.Size([batch_num, 512, 38, 38])
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


class DBox(object):
    """class outputs default boxes"""
    def __init__(self, cfg):
        super(DBox, self).__init__()

        self.image_size = cfg['input_size']
        # size of feature map for each source
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']   # pixel sizes of default box
        self.min_sizes = cfg['min_sizes']  # pixel sizes of smaller squares
        self.max_sizes = cfg['max_sizes']  # pixel sizes of bigger squares
        self.aspect_ratios = cfg['aspect_ratios']  # aspect ratio of rectangle default box

    def make_dbox_list(self):
        """create default box"""
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # image size of feature
                f_k = self.image_size / self.steps[k]

                # center coordinates of default box normalized with range from 0 to 1.
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # smaller default box [cx,cy, width, height] with aspect ratio 1.
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # bigger default box [cx,cy, width, height] with aspect ratio 1.
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # default box [cx,cy, width, height] with other aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # convert default box into tensor ( torch.Size([8732, 4]) )
        output = torch.Tensor(mean).view(-1, 4)

        # to prevent default box from sticking out of image.
        output.clamp_(max=1, min=0)

        return output


def decode(loc, dbox_list):
    """transform default box into bounding box using offset.

    Args:
    - loc: offset to perform inference with SSD model, [8732,4]
    - dbox_list: default box, [8732,4]

    Returns:
    - boxes: bounding box, [xmin, ymin, xmax, ymax]
    """
    # default box : [cx, cy, width, height]
    # loc : [Δcx, Δcy, Δwidth, Δheight]

    # calculate bounding box from offset
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    # size of boxes is torch.Size([8732, 4])

    # convert coordinate of bounding box from [cx, cy, width, height] to [xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """Non-Maximum Suppression that deletes overlapped boxes.

    Args:
    - boxes: bounding box
    - scores: confidence

    Returns:
    - keep: list of index that passed nms in descending order
    - count: number of bounding box passed nms
    """

    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    # calculate area of each bounding box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # score in ascending order
    v, idx = scores.sort(0)

    # take indexes of top k bouding boxes.
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w*tmp_h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union

        idx = idx[IoU.le(overlap)]

    return keep, count


class Detect(Function):
    """output bounding boxes from conf and loc in inference"""
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """calculate forward

        Args:
        - loc_data: offset [batch_num,8732,4]
        - conf_data: confidence of detection [batch_num, 8732,num_classes]
        - dbox_list: default box [8732,4]

        Returns:
        - output: torch.Size([batch_num, 21, 200, 5])
        """

        num_batch = loc_data.size(0)
        # num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        # normalize conf with softmax
        conf_data = self.softmax(conf_data)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # re-order cof_data from [batch_num,8732,num_classes] to [batch_num,num_classes,8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):

            decoded_boxes = decode(loc_data[i], dbox_list)

            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):

                c_mask = conf_scores[cl].gt(self.conf_thresh)

                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference
        self.num_classes = cfg["num_classes"]

        # build SSD network
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        # create default box
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == 'inference':
            params = {}
            if "conf_thresh" in cfg:
                params["conf_thresh"] = cfg["conf_thresh"]
            if "top_k" in cfg:
                params["top_k"] = cfg["top_k"]
            if "nms_thresh" in cfg:
                params["nms_thresh"] = cfg["nms_thresh"]
            self.detect = Detect(**params)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # loc: torch.Size([batch_num, 34928])
        # conf: torch.Size([batch_num, 183372])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc: torch.Size([batch_num, 8732, 4])
        # conf: torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])
        else:
            return output


class MultiBoxLoss(nn.Module):
    """SSD loss function"""
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super().__init__()
        # threshold of jaccard coefficient of match function
        self.jaccard_thresh = jaccard_thresh
        # Hard Negative Mining ratio
        self.negpos_ratio = neg_pos
        self.device = device

    def forward(self, predictions, targets):
        """calculate loss function

        Args:
        - predictions: output of SSD net in training
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])。
        - targets: [num_batch, num_objs, 5]

        Returns:
        - loss_l: loss of loc
        - loss_c : loss of conf
        """

        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.size(0)  # size of mini-batch
        num_dbox = loc_data.size(1)  # number of DBox
        num_classes = conf_data.size(2)  # number of classes

        # label of the correct BBox closest to each DBox
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        # position of the correct BBox closest to each DBox
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)

            dbox = dbox_list.to(self.device)

            variance = [0.1, 0.2]
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)

        # loss of position
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # transform pos_mask into size of loc_data
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # loss of class prediction
        batch_conf = conf_data.view(-1, num_classes)

        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        N = num_pos.sum().float()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
