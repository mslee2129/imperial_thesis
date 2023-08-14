#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=2:mem=48gb:ngpus=2:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/train.py --dataroot ./data/nmed-t-prep --name cgan_resnet_9_batch_16_layers_4 --gpu_ids 0,1 --model pix2pix --netG resnet_9blocks  --dataset_mode supervised --display_id -1 --input_nc 1 --output_nc 1 --batch_size 16 --n_layers_D 4 --netD n_layers