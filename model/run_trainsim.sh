#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ttrain_per
#SBATCH -o ttrain_per.%J.out
#SBATCH -e ttrain_per.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH --time=00:25:00
#SBATCH --mem=10G

printf 'sim_test,batchsize=8'
python train_perpr.py