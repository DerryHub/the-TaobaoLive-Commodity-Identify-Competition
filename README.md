# Object Detection
## Configuration File
```python
from config import get_args_efficientdet
opt = get_args_efficientdet()
print(opt)
```
## Data Preparation
```python
from process_data import processTrain, processValidation
label = {}
label['label2index'] = {}
label['index2label'] = {}
processTrain(label)
processValidation(label)
```

data from [images, videos] is in path data/[train, validation]\_[images, videos] and data/[train, validation]\_[images, videos]\_annotation.json

[train, validation]\_[images, videos]\_annotation.json
```json
{
    "annotations": [
        {
            "img_name": "{item id}_{frame}.jpg",
            "annotations": [
                {
                    "label": 22,
                    "viewpoint": 0,
                    "display": 1,
                    "instance_id": 0,
                    "box": [221, 507, 614, 1039]
                },
                ...
            ]
        },
        ...
    ]
}
```
## Dataset
```python
from dataset import EfficientdetDataset
dataset = EfficientdetDataset(root_dir, mode, imgORvdo, transform, maxLen, PAD)
```
## EfficientDet
```python
from config import get_args_efficientdet
from efficientdet.efficientdet import EfficientDet
opt = get_args_efficientdet()
model = EfficientDet(opt)
```
## Train EfficientDet
```shell
python train_efficientdet.py
```

## Validation EfficientDet
```shell
python validation_efficientdet.py
```

# Match
## Configuration File
```python
from config import get_args_arcface
opt = get_args_arcface()
print(opt)
```
## Data Preparation
```python
from process_data import processTrain, processValidation
saveNumpyInstance('data', 'train', (256, 256))
saveNumpyInstance('data', 'validation', (256, 256))
createInstance2Label('data')
createInstanceID()
```

the shape of matrix in data/[train, validation]\_instance is $256\times256\times3$

the input of arcface is $3\times224\times224$

## Dataset
### Arcface
```python
from dataset import ArcfaceDataset
dataset = ArcfaceDataset(root_dir, mode, size, flip_x, maxLen, PAD, imgORvdo)
```

### Triplet Loss
```python
from dataset import TripletDataset
dataset = TripletDataset(root_dir, mode, size, flip_x)
```

### Hard Triplet Loss
```python
from dataset import HardTripletDataset
dataset = HardTripletDataset(root_dir, mode, size, flip_x, n_samples)
```

## Arcface
### Backbone
```python
from config import get_args_arcface
from arcface.resnest import ResNeSt
opt = get_args_arcface()
backbone = ResNeSt(opt)
```
### Head
```python
from config import get_args_arcface
from arcface.head import Arcface
opt = get_args_arcface()
head = Arcface(opt)
```

## Train Match Model
### Arcface
```shell
python train_arcface.py
```

### Triplet Loss
```shell
python train_triplet.py
```

### Hard Triplet Loss
```shell
python train_hardtriplet.py
```

## Validate Match Model
```shell
python validation_arcface.py
```

# Validate with both object detection and match model
```shell
python validation_all.py
python cal_score.py
```

# Docker
```shell
cd docker
```
## Build
```shell
docker build -t image_name .
```
## Push
```shell
docker push image_name
```

## Test
```shell
nvidia-docker run -v /data/validation_dataset_part1:/tcdata/test_dataset_3w your_image sh run.sh
```