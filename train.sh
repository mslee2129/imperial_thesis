#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=2:ncpus=4:mem=24gb:ngpus=4:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/train.py --dataroot ./data/nmed-t-prep/AB --name pix2pix_psd --gpu_ids 0,1,2,3,4,5,6,7 --model pix2pix --netG unet_128 --netD basic --norm batch --dataset_mode aligned batch_size 32 --gan_mode vanilla --display_id -1 
