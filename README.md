# OD
Empowering Lightweight Detectors: Orientation Distillation via Anti-ambiguous Spatial Transformation for Remote Sensing Images
## Install
  - Our codes are based on [MMRotate](https://github.com/open-mmlab/mmrotate). Please follow the installation of MMRotate and make sure you can run it successfully.
  - This repo uses mmdet==0.3.4+
  
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmrotate's codes.
  - Add the mmrotate/. in our codes to the mmrotate/ in mmrotate's codes.
  
## Train
```
#single GPU
python tools/train.py configs/od/rotated_retinanet/rotated_retinanet_obb_r18_r101_fpn_1x_dota_le90.py --gpus 1

#multi GPU
bash tools/dist_train.sh configs/od/rotated_retinanet/rotated_retinanet_obb_r18_r101_fpn_1x_dota_le90.py 8
```

## Test

```
#single GPU
python tools/test.py configs/od/rotated_retinanet/rotated_retinanet_obb_r18_r101_fpn_1x_dota_le90.py $new_mmdet_pth --eval bbox

#multi GPU
bash tools/dist_test.sh configs/od/rotated_retinanet/rotated_retinanet_obb_r18_r101_fpn_1x_dota_le90.py $new_mmdet_pth 8 --eval bbox
```
Subsequently, we submitted the results of the trained model to the official website for testing.

## Trained model and log file
  - Trained model can be finded in [GoogleDrive]( https://pan.baidu.com/s/1UzCgvobiv9mlHZqP8qnIfQ?pwd=uwj3).
  - Log file can be finded in workdirs/.
