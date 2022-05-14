# YOLOV3：You Only Look Once in Pytorch
- This implementation changes the shape of anchor boxes from rectangles to circles

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#references'>References</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Changes
- Change the prior anchor shape from rectangles to circles (with different scale of radius) /model_data/yolo_anchors.txt
- Change the function of calculate IoU /nets/yolo_training.py
- Change the function of calculate gIoU /nets/yolo_training.py
- Change the preprocess of dataset (change rectangles of bbox to circle)  /utils/dataloader.py


## Installation
  ```Shell
  pip install -m requirements.txt
  ```

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

- same as https://github.com/JethroLee98/SSD_pytorch_circle

## Training YOLOv3
- First download the pre-trained YOLOv3 PyTorch base network weights at:              https://drive.google.com/file/d/1mZrJvIZqv1oVkHvs-swNjjsMWPhtUh5_/view?usp=sharing
- By default, we assume you have downloaded the file in the `/model_data` dir:

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python voc_annotation.py
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.


## Evaluation
To evaluate a trained network:

```Shell
python get_map.py
```

You can specify the parameters listed in the `get_map.py` file by flagging them or manually changing them.  


## Performance

#### VOC2007 Test

##### mAP (threshold set as 0.3, since we change the shape to circle which will reduce the normal IoU between rectangles)

| Original | Change Anchor shape to circle |
|:-:|:-:|
| 67.2 % | 32.12% |

- AP for aeroplane = 57.59%
- AP for bicycle = 44.09%
- AP for bird = 29.22%
- AP for boat = 15.74%
- AP for bottle = 4.49%
- AP for bus = 46.53%
- AP for car = 39.68%
- AP for cat = 50.30%
- AP for chair = 14.36%
- AP for cow = 13.04%
- AP for diningtable = 44.19%
- AP for dog = 38.74%
- AP for horse = 30.64%
- AP for motorbike = 42.81%
- AP for person = 17.28%
- AP for pottedplant = 6.74%
- AP for sheep = 4.35%
- AP for sofa = 44.91%
- AP for train = 61.85%
- AP for tvmonitor = 35.81%

## Demos

### Use a pre-trained YOLOv3 network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest YOLO model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * YOLOv3 trained on VOC0712 (newest PyTorch weights)
      - https://drive.google.com/file/d/1CWLh-2xHC5Z6pPzPQfYnGEf4mzpKXUel/view?usp=sharing
    * YOLOv3 trained on VOC0712 (change anchor shape to circle)
      - https://drive.google.com/file/d/12Z2zaxmkLMNcJRp-5CilRaH_wqxRifFM/view?usp=sharing
#### Demo
Change paramters in yolo.py
```Shell
python predict.py
input img/street.jpg
```
      


## References
- https://github.com/alexweissman/geometry
- https://github.com/bubbliiiing/yolo3-pytorch
