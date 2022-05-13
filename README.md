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
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

Params used: 
- batch size = 64
- learning rate = 1e^-5

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


## Performance

#### VOC2007 Test

##### mAP

| Original | Change Anchor shape to circle |
|:-:|:-:|
| 67.2 % | 31.46% |

- AP for aeroplane = 0.0113
- AP for bicycle = 0.5812
- AP for bird = 0.4209
- AP for boat = 0.1370
- AP for bottle = 0.0305
- AP for bus = 0.2779
- AP for car = 0.1962
- AP for cat = 0.5656
- AP for chair = 0.2505
- AP for cow = 0.3047
- AP for diningtable = 0.2324
- AP for dog = 0.4635
- AP for horse = 0.5791
- AP for motorbike = 0.5178
- AP for person = 0.2800
- AP for pottedplant = 0.0991
- AP for sheep = 0.2820
- AP for sofa = 0.1651
- AP for train = 0.4641
- AP for tvmonitor = 0.4337

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (change anchor shape to circle)
      - https://drive.google.com/file/d/1i4Lg_Y-BIFv8kFBd5QzYSL4E39Cz5wrY/view?usp=sharing
      

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## References
- https://github.com/alexweissman/geometry
- https://github.com/bubbliiiing/yolo3-pytorch
