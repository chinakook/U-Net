# U-Net-MX
**This is an reimplementation for MXNet of [U-Net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net).**
See the following references for more information:
```
"U-Net: Convolutional Networks for Biomedical Image Segmentation."
Olaf Ronneberger, Philipp Fischer, and Thomas Brox
arXiv preprint arXiv:1505. 04597, 2015.
```
[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

## Usage

### Data preparation
  Prepare source images in './data/train' and groud truth images in './data/train_label' following below requirements:
  1. Source images should be RGB or gray 8-bit images.
  2. Ground truth image should be named same to corresponding source image(ignore extension name).
  3. Ground truth images and source images should have all the same height and width.
  4. Ground truth image should be labeled as: Background - 0, Foreground - 1.

### Training
  Training with standard softmax cross entropy loss as follow:
  ```
  python ./unet_train.py
  ```
  Training with softmax cross entropy loss with class weights as follow:
  ```
  python ./unet_train_class_weighted.py
  ```
### Testing
  Change the image path to your own one in line 19 of unet_test.py, and simply run:
  ```
  python ./unet_test.py
  ```


## Requirements
  1. The latest [MXNet](https://github.com/dmlc/mxnet) with gluon API.
  2. Any NVIDIA GPUs with at least 2GB memory should be OK.

## Differences
  1. Our reproduction make the output shape be same to the data shape.
  2. The data shape can be variant.
  
## TODO:
- [ ] Test demo
- [ ] Train demo
- [ ] Train from vgg model
- [ ] Multi-class support
- [ ] Resum training
- [ ] Automatic class weights statistic
