#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 4932_protein_test1
#SBATCH -o 4932_protein_test1.%J.out
#SBATCH -e 4932_protein_test1.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=END
#SBATCH --time=01:30:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:2

python 0esmExtract.py 