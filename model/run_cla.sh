#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J class_1024_40
#SBATCH -o class_1024_40.%J.out
#SBATCH -e class_1024_40.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2-15:30:00
#SBATCH --mem=100G

printf "batchsize = 1024 2-1.15min epoch=40 ~12h-15h"
python classify.py --batchsize 1024 --epchonum 40