#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=24gb

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/model/datasets/combine_A_and_B.py --fold_A ./data/nmed-t-prep/A --fold_B ./data/nmed-t-prep/B --fold_AB ./data/nmed-t-prep/AB