#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J train_f_2dot
#SBATCH -o train_f_2dot.%J.out
#SBATCH -e train_f_2dot.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=03:25:00
#SBATCH --mem=20G

printf 'dot product with batch trainning...\n'
printf '>>--batch-size  32 --epochs 50\n'
printf '>>model on ONE feature individually NO UPDATE\n'
printf '>>lr 1e -4 '
printf '>>using Relu()'
printf 'traindotfeature_0206 multi.py'
printf '################'
printf 'only one fold of all'
printf '################'

# python traindotfeature_0206.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 30 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twof -lr 1e-3
#twof with relu works
python traindotfeature_0206multi_metric.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 30 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twof -lr 1e-3 --metric dot 
