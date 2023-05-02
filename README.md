# Paper study and implementation - CS6450 Visual computing

# Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization
by [Ziqi Zhou](https://zzzqzhou.github.io/), [Lei Qi](http://palm.seu.edu.cn/qilei/), [Xin Yang](https://xy0806.github.io/), Dong Ni, [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm). 

## Introduction

This repository is for paper study and implementation of paper https://arxiv.org/abs/2112.11177.
The original code is available at https://github.com/zzzqzhou/Dual-Normalization.
Additional files were added to do preprocessing and testing improvelents which are listed in Preprocessing, Training and Testing section below.

#### Created/Modified file list:
<ul>
<li>preprocess_func_cardiac_test_v2.py 
<li>preprocess_func_cardiacv2.py
<li>preprocess_func_brats_test.py
<li>preprocess_func_nii.py
<li>test_dn_unet_brats_ss.py
<li>test_dn_unet_brats_sd.py
<li>test_dn_unet_ct.py
</ul>

## Data Preparation

### Dataset
I tried model generation on following two datasets:
[BraTS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html) | [MMWHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) 

BraTS 2018 dataset is not available from the original source, so had to download from Kaggle. The data was in slightly different format (nii instead of npz) so preprocessing code was updated to handle this.


### File Organization

T2 as source domain
``` 
├── [Your BraTS2018 Path]
    ├── npz_data
        ├── train
            ├── t2_ss
                ├── sample1.npz, sample2.npz, xxx
            └── t2_sd
        ├── test
            ├── t1
                ├── test_sample1.npz, test_sample2.npz, xxx
            ├── t1ce
            └── flair
```


## Preprocessing, Training and Testing
### Preprocessing:
#### BraTS 2018 dataset:

If you are using the dataset downnloaded from Kaggle, use following script
```
 python preprocess_func_nii.py
```
For preprocessing testing data from BraTS 2018 dataset
```
python preprocess_func_brats_test.py
```
Note each of the script takes these 3 parameters. Please update the paths and modality as applicable:
```
    data_root = '/mnt/disks/vc_data/dataset/brats2018/MICCAI_BraTS_2018_Data_Training/HGG'
    target_root = '/mnt/disks/vc_data/dataset/brats2018/MICCAI_BraTS_2018_Data_Training/MICCAI_BraTS_2018_Data_Testing_processed'
    modality = 'flair'
```

#### Cardiac dataset:

Training data
```
 python preprocess_func_cardiacv2.py
```

Testing data
```
python preprocess_func_cardiac_test_v2.py
```

Note each of the script takes these 4 parameters. Please update the paths, modality and sub_modality as applicable. Sub modality is newly introduced. Please refer to the http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/data.html website for various segmentation. For e.g. 500 is the label for the left ventricle blood cavity.
```
    data_root = '/mnt/disks/vc_data/dataset/cardiac/ct_train'
    target_root = '/mnt/disks/vc_data/dataset/cardiac/ct_train_processed_v2'
    modality = 'ct'
    sub_modality = 500
```

### Training
Train on source domain T2.
```
python -W ignore train_dn_unet.py \
  --train_domain_list_1 t2_ss --train_domain_list_2 t2_sd --n_classes 2 \
  --batch_size 16 --n_epochs 50 --save_step 10 --lr 0.001 --gpu_ids 0 \
  --result_dir ./results/unet_dn_t2 --data_dir [Your BraTS2018 Path]/npz_data
```

### Testing
For testing I have created 2 separate scripts for source similar and source dissimilar domains.
Please refer to presentation for the details of which test domains belong to which category and use appropriate testing script.
Test on target domains (T1, T1ce and Flair).

#### For source similar domains
```
python -W ignore test_dn_unet_brats_ss.py \
  --test_domain_list t1 t1ce flair --model_dir ./results/unet_dn_t2/model
  --batch_size 32 --save_label --label_dir ./vis/unet_dn_t2 --gpu_ids 0 \
  --num_classes 2 --data_dir [Your BraTS2018 Path]/npz_data
```

#### For source dis-similar domains
```
python -W ignore test_dn_unet_brats_sd.py \
  --test_domain_list t1 t1ce flair --model_dir ./results/unet_dn_t2/model
  --batch_size 32 --save_label --label_dir ./vis/unet_dn_t2 --gpu_ids 0 \
  --num_classes 2 --data_dir [Your BraTS2018 Path]/npz_data
```
