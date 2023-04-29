#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ttest
#SBATCH -o ttest.%J.out
#SBATCH -e ttest.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH --time=00:25:00
#SBATCH --mem=10G

printf 'test,batchsize=8'
python test.py