# platform-template-object-detection

Template of object detection for ABEJA Platform.

## how to use

Clone this repository : 

```
git clone https://github.com/abeja-inc/platform-template-object-detection/
cd platform-template-object-detection
```

Install Visdom and run server : 

```
pip install visdom
python -m visdom.server
```

## Datasets

### COCO 

sorry, on construction.


### VOC 

Downloda dataset :

```
sh detection/Models/SSD/data/scripts/VOC2007.sh
```
and
```
sh detection/Models/SSD/data/scripts/VOC2012.sh
```

And you shold move some VOC directories, like this :

```
SSD/
├── VOC
│   ├── VOCdevkit
│       ├── VOC2007
│       │   ├── ...
│       │   ├── ...
│       ├── VOC2012
│           ├── ...
│           ├── ...
├── train.py
├── eval.py
.
.
.
```


## Train SSD Model

```
python detection/Models/SSD/train.py --dataset <VOC or COCO> \
                                     --dataset_root <path to your saved data> \
                                     --cuda <0 or 1>
```

For example, you want to train with VOC20012 and without cuda : 

```
python train.py --dataset 'VOC' --dataset_root ./VOC/VOCdevkit/ --cuda 0
```

## Evaluatioin

```
python detection/Models/SSD/eval.py
```
