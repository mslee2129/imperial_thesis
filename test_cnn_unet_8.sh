#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=24gb:ngpus=1

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/test.py --dataroot ./data/nmed-t-prep --name cnn_unet_batch_8 --input_nc 1 --output_nc 1 --model pix2pix --results_dir cnn_res --netG unet_128 --norm instance --dataset_mode supervised