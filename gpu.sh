#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1:gpu_type=RTX6000

cd $PBS_O_WORKDIR

nvidia-smi -q -x

