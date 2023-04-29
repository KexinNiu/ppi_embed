#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J min_esm
#SBATCH -o min_esm.%J.out
#SBATCH -e min_esm.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH --time=01:20:00
#SBATCH --mem=10G

# printf "minhash eval esm1&esm2 klist"
printf "minhash eval prott5 klist"

python minhashlsh.py
