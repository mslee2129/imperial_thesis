#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=24gb:ngpus=1:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/test.py --dataroot ./data/nmed-t-prep --name cgan_resnet_9_batch_8_layers_5 --input_nc 1 --output_nc 1 --model pix2pix --results_dir cgan_res --netG resnet_9blocks --norm instance --dataset_mode supervised
