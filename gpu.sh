#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=4gb:ngpus=1:gpu_type=RTX6000

cd $PBS_O_WORKDIR

nvidia-smi -q -x

