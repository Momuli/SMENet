# SMENet
This is a pytorch implementation of SMENet

## Requirements
1. pytorch == 1.1.0

2. cuda 8.0

3. python == 3.7

4. [opencv(CV2)](https://pypi.org/project/opencv-python/)

## Data Prepare
1. Please download NWPU VHR-10
2. Convert to PASCAL VOC data format
3. Create dataset folder
```
./SMENet/VOCNWPU/
```
4.Data format
```
├── VOCNWPU
│   ├── VOC2012
│       ├── Annotations
│       ├── JPEGImages
│       ├── ImageSets
│         ├── train.txt
│         ├── test.txt
|         ├── val.txt
|         ├── trainval.txt
```

## Demo

1.Please download weights file `SMENet.pth`, and put it to:
```
./SMENet/weights/
```
2. Run `visual_SMENet.py`:
```
cd ./SMENet/demo/visual_SMENet.py
modify parser.add_argument('--trained_model', default='../weights/SMENet.pth', type=str, help='Trained state_dict file path to open')
python visual_SMENet.py
```
## Train
if you want to train your own dataset:
```
1. Convert your dataset to PASCAL VOC and put it to `./SMENet/dataset-file-name/`
2. Modify parameters  `HOME` and  `num_classes` in `./SMENet/data/config.py` :
    HOME= absolute path of the SMENet folder
3. Modify parameters `VOC_CLASSES` in `./SMENet/data/voc0712.py`
4. `python train_SMENet.py`
5. save `*.pth` to weights, like `./SMENet/weights/*.pth`
```
## Eval
if you want to eval trained model:
```
1. cd ./SMENet/eval.py
2. Modify parser.add_argument('--trained_model', default='weights/*.pth', type=str, help='Trained state_dict file path to open')
3. python eval.py
```
