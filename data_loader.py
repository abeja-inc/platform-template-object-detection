import torch


def od_collate_fn(batch):
    """
    annotation data size is different among images depending on detected objects.
    customized collate_fn is defined to create DataLoader that can handle this difference.
    collate_fn is a function that create mini-batch from list of dataset.
    this function transform the batch appending mini-batch index.
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] is image
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] is annotation ( or `ground truth` )

    # imgs is mini-batch size of list and each element is torch.Size([3, 300, 300]).
    # transform this list into torch.Size([batch_num, 3, 300, 300]) tensor.
    imgs = torch.stack(tuple(imgs), dim=0)

    # targets is mini-batch size of list of ground truth.
    # each element of list is [n, 5] where n is the number of detected objects,
    # and is different among images.
    # 5 means [xmin, ymin, xmax, ymax, class_index]

    return imgs, targets
