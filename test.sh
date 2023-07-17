#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=24gb:ngpus=1

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/test.py --dataroot ./data/nmed-t-prep/AB --name pix2pix_AtoB --model pix2pix --results_dir pix2pix_res
