# Platform Template Object Detection

This is the template of object-detection task for ABEJA Platform.

## Requirements

- Python 3.6.x
- [For local] Install [ABEJA Platform CLI](https://developers.abeja.io/developer-tools/cli/)

## Conditions
- Transfer learning from VGG16 ImageNet.
- Allow only 1 category with multiple labels
- Allow up to two dataset
    - with two dataset, either of them should be aliased with `val`.

## Parameters

| env | type | description |
| --- | --- | --- |
| BATCH_SIZE | int | Batch size. Default `32`. |
| EPOCHS | int | Epoch number. This template applies "Early stopping". Default `50`. |
| IMG_SIZE | int | Image size. **Currently only SSD300 (size=300) is supported!!** Automatically resize to this size. Default `300`. |
| SHUFFLE | bool | Shuffle train dataset. Default `true`. |
| RANDOM_SEED | int | Random seed. If set, use it for a data shuffling. Default `None`. |
| MAX_ITEMS | bool | Number of items to use. Default `None` which means use all items |
| TEST_SIZE | float | Ratio of test dataset. Default `0.4` |
| LEARNING_RATE | float | Learning rate. Need to be from `0.0` to `1.0`. Default `0.001`. |
| MOMENTUM | float | Momentum factor. Need to be from `0.0` to `1.0`. Default `0.0`. |
| WEIGHT_DECAY | float | Weight decay (L2 penalty). Need to be from `0.0` to `1.0`. Default `0.0`. |
| DAMPENING | float | Dampening for momentum. Need to be from `0.0`. Default `0.0`. |
| NESTEROV | float | Enables Nesterov momentum. Default `False`. |
| CONF_THRESHOLD | float | Confidence threshold to filter out bounding boxes with low confidence. Default `0.01`. |
| TOP_K | int | Number of bounding boxes to be taken. Default `200`. |
| NMS_THRESHOLD | float | The threshold for IoU to consider bounding boxes as the same. Default `0.45`. |
| OVERLAP_THRESHOLD | float | The overlap threshold used when matching boxes. Need to be from `0.0` to `1.0`. Default `0.5` |
| NEG_POS | int | Hard Negative Mining ratio. Default `3` |
| CONFIDENCE_THRESHOLD | float | Results above this threshold will be returned. **This option is valid only for prediction (or inference).** Default `0.1` |

## Run on local

You can run on local using [ABEJA Platform CLI](https://developers.abeja.io/developer-tools/cli/)
You need to setup your [configuration](https://developers.abeja.io/reference/cli/configuration-command/init/) at first.


### Debug on local

You can train on local with [debug-local](https://developers.abeja.io/reference/cli/training-command/training-debug-local/) command.

```
$ abeja training debug-local \
  --handler train:handler \
  --image abeja-inc/all-cpu:19.04 \
  --organization_id xxxxx \ 
  --datasets train:xxxxx
```


## Train on ABEJA Platform

### Prepare `training.yaml` file

```
name: my-detection-model-from-template
handler: train:handler
image: abeja-inc/all-gpu:19.04
datasets:
  "train": "xxx"
  "val": "xxx"
```

### Create Job definition

Create job definition on the platform at first.

```
abeja training create-job-definition
```

### Create Version & Create Job

When you modify your code, run the command to deploy your code on the platform.

```
abeja training create-version
```

After deploying it, you can create the job by running the command below.

```
abeja training create-job -d "my first job"
```

After creating job, you can check the status by ABEJA Platform Console.

### Inference on local

You can do prediction on local with [run-local](https://developers.abeja.io/reference/cli/model-command/run-local/) command.

```
$ abeja model run-local \
  --handler predict:handler \
  --image abeja-inc/all-cpu:19.04 \
  --environment TRAINING_JOB_DATASETS:'{"data":1111111111111}' \
  --environment CONFIDENCE_THRESHOLD:0.9 \
  --input sample.jpg
```

**environment variables for inference**

| env | type | description |
| --- | --- | --- |
| TRAINING_JOB_DATASETS | str | Dataset IDs. e.g `'{"data":1111111111111}'`.  The prediction script fetches the label information from training data. |

You can see more detail of the parameters on `parameters.py`.

Also you can start inference API on local with [run-local-server](https://developers.abeja.io/reference/cli/model-command/run-local-server/) command.

```
$ abeja model run-local-server \
  --handler predict:handler \
  --image abeja-inc/all-cpu:19.04 \
  --environment TRAINING_JOB_DATASETS:'{"data":1111111111111}' \
  --environment CONFIDENCE_THRESHOLD:0.3 \
```
