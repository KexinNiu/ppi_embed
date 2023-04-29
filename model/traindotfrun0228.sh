#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ttrain_gene
#SBATCH -o ttrain_gene.%J.out
#SBATCH -e ttrain_gene.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH --time=0:59:00
#SBATCH --mem=20G

printf 'generative...\n'
printf 'number = 5 \n'
printf '>>model on two feature individually\n'
printf '>>non UPDATE FOR TRAINNING EMBEDDING\n'
printf 'traindotfeature_0228.py \n'

python traindotfeature_0228.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 16 --epochs 20 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname generative  -lr 1e-2 