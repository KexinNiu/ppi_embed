#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J tt_tf_64_e4
#SBATCH -o tt_tf_64_e4.%J.out
#SBATCH -e tt_tf_64_e4.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=10:59:00
#SBATCH --mem=20G

printf 'traindotfeature_0219.py '
printf 'dot product with batch trainning...\n'
printf '>>--batch-size 64 --epoch100 \n'
printf '>>model on two feature individually\n'
# printf '>>UPDATE FOR TRAINNING EMBEDDING\n'

python traindotfeature_0219.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 100 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twof  -lr 1e-4

# python traindotfeature_0219.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 16 --epochs 50 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twofupdate  -lr 1e-2
# python traindotfeature_0219.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 16 --epochs 50 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname generative  -lr 1e-4 