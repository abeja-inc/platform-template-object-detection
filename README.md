# ABEJA Platform Data Loader

This is small class for uploading and downloading datalake or dataset.

## How to use

First, clone this repository:

```
$ git clone https://github.com/abeja-inc/platform-template-object-detection/
```

Then, move to this repository, and import this class.

```
$ cd platform-template-object-detection/
```

Run REPL:

```
>>> from load_data import DataLoader
>>> ID = <your organization ID of abeja platform>
>>> dataloader = DataLoader(organization_id=ID)
```

Here you can do some actions: 
- create datalake channel
- upload datalake
- create dataset
- upload dataset
- download dataset

In detail, check docstrings in `load_data.py`.