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

COCO : 

```
sh detection/Models/SSD/data/scripts/COCO2014.sh
```

VOC : 

```
sh detection/Models/SSD/data/scripts/VOC2007.sh
```
or
```
sh detection/Models/SSD/data/scripts/VOC2012.sh
```

## Train SSD Model

```
python detection/Models/SSD/train.py
```

## Evaluatioin

```
python detection/Models/SSD/eval.py
```