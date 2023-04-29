#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J eval_1
#SBATCH -o eval_1.%J.out
#SBATCH -e eval_1.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=00:25:00
#SBATCH --mem=2G

printf 'evaluation...\n'

# printf '#####onef######'

# printf 'python eval_ --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 50 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname onef -lr 1e-4 --metric dot'
# python eval_0206multi_metric.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 100 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname onef  --metric dot --load

# printf '__________________________________________'
# printf '######twof######'
# printf '######twof######'
# printf '######twof######'
# printf '######twof######'
# printf '######twof######'
# printf '######twof######'
printf '######twof######'
# printf '######twof######'
python eval_0206multi_metric.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 100 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname twof  --metric dot --load
