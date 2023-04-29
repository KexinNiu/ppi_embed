#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J train_rk0408_1
#SBATCH -o train_rk0408_1.%J.out
#SBATCH -e train_rk0408_1.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=09:35:00
#SBATCH --mem=10G

printf 'PULOSS...\n'
printf '>>using Relu()'
printf 'Based on 0206 metric.py'
printf '################onef -lr 1e-4 --metric dot'
printf '################epoch = 100'


python traindotfeature_0206_rk.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 100 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname onef -lr 1e-3 --metric dot
# python traindotfeature_0206_rk.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 5 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname onef -lr 1e-3 --metric dot
