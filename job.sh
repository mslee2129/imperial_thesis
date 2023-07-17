#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=2:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/train.py --model pix2pix --dataroot ./data/nmed-t-prep/AB  --dataset_mode aligned --no_html --netG unet_256 --netD basic --gan_mode vanilla --name pix2pix_AtoB --batch_size 32 --gpu_ids 1 --display_id -1