#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=2:mem=24gb

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/train.py --dataroot ./data/nmed-t-p2p/ --name cnn_psnr --gpu_ids -1--model cnn --netG resnet_9blocks --dataset_mode aligned --display_id -1 --input_nc 1 --output_nc 1 --norm batch
