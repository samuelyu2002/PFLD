# PFLD

## Install Requirements
```
pip3 install -r requirements.txt
```
## Dataset Download
### 300W Dataset
The 300W Dataset consists of 3148 training images and 689 testing images. All the separate components of the dataset can be downloaded [here](https://ibug.doc.ic.ac.uk/resources/300-W/). The training set consist of the entire AFW dataset, and the training set for LFPW and HELEN. The common testing set includes the testing sets of LFPW and HELEN, and the challenging set consists of the iBUG dataset. Together, they make the full testing dataset. 

To prepare the dataset, place all the train images in data/300W/train, and all the test images in data/300W/test.
```
$ cd data
$ python3 set_preparation_68.py
```
### WFLW Dataset
The WFLW Dataset consists of 7500 training images and 2500 testing images. It can be downloaded [here](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA). 

To prepare the dataset, simply place WFLW_images folder within the WFLW folder. Then, run:
```
$ cd data
$ python3 set_preparation_98.py
```
### Demo
For a simple demo, run demo.py or camera.py. You can specify which image to use for demo.py with the argument --image_path.

### Training
Please specify --dataroot and --val_dataroot to link to the list.txt file containing the train and test annotations. For a 68-point dataset such as 300W, --num_landmark should be 68, while for a 98-point dataset such as WFLW, --num_landmark should be 98. You can change the backbone to either MobileNet or VoVNet with the --backbone argument. For simple training, create a folder named "1" within checkpoint, and run:
```
$ python3 train.py
```

### Testing
Please specify the arguments based on which model you want to test. There is a pretrained MobileNet backbone model provided. To test it, simply run:
 ```
$ python3 test.py
```

### Reference
PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf

Original Pytorch Implementation: https://github.com/polarisZhao/PFLD-pytorch

Tensorflow Implementation: https://github.com/guoqiangqi/PFLD
