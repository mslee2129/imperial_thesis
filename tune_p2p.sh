#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=2:mem=100gb:ngpus=2:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/train.py --dataroot ./data/nmed-t-p2p/AB --name pix2pix_tune --gpu_ids 0,1 --model pix2pix --netG resnet_9blocks --netD basic --dataset_mode aligned --batch_size 32 --gan_mode lsgan --display_id -1 --input_nc 1 --output_nc 1
