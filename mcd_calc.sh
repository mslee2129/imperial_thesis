#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=24gb

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/venv/bin/activate

cd $PBS_O_WORKDIR

python ./code/mcd_calc.py