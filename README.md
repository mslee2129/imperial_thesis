# Imperial Thesis

## About
This is a GitHub repository for my Individual Project for the MSc Computing degree at Imperial College London. This project attempts to reconstruct musical stimuli from EEG signals using generative models. The report can be seen below.

## Dataset
The dataset used for training can be found [here](https://exhibits.stanford.edu/data/catalog/jn859kj8079) (NMED-T) and [here](https://openneuro.org/datasets/ds002721/versions/1.0.2) (Film Music).
Save the dataset in ~/data/


## Preprocessing
There are two preprocessing pipelines. For preprocessing the Film Music dataset from scratch, use the MATLAB scripts in code/preprocessing/matlab. For preprocessing and segmenting the NMED-T dataset for training preparation, use files in code/preprocessing/python. The preprocessed data should be saved under data/nmed-t-prep/

## Models
There are three models that can be trained: CNN, cGAN and cCGAN. These can be found in code/model/models

## Training/Testing
Training script example:
```
python ./code/model/train.py --dataroot ./data/nmed-t-prep --name cgan_resnet_9_batch_8 --gpu_ids 0 --model pix2pix --netG resnet_9blocks  --dataset_mode supervised --display_id -1 --input_nc 1 --output_nc 1 --batch_size 8 --netD basic
```

Set 
``` 
--label_smoothing
```
to traing with label smoothing
## Citation
The code for the GANs has been largely inspired from the original [pix2pix/cycleGAN paper](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
