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
