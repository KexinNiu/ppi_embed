#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J tt_64_twof_e4
#SBATCH -o tt_64_twof_e4.%J.out
#SBATCH -e tt_64_twof_e4.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=10:25:00
#SBATCH --mem=20G

printf 'dot product with batch trainning...\n'
printf '>>--batch-size  64 --epochs 30\n'
printf '>>model on two feature individually NO UPDATE\n'
printf '>>lr 1e-3'
# printf '>>using Relu()'
printf 'traindotfeature_0206.py'
printf 'result save'

python traindotfeature_0206.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 30 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twof -lr 1e-3
#twof with relu works
