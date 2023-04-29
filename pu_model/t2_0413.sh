#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J t0413_2
#SBATCH -o t0413_2.%J.out
#SBATCH -e t0413_2.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=06:05:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

printf 'PULOSS...\n'
printf '>>using sigmoid()\n'
printf '>>two\n'

printf '################twof -lr 1e-4 --metric dot\n'
printf '################twof epoch 100 '

# module load esm/1.0.3 
# module load machine_learning/2021.09 
python train0413.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/data/4932 --batch-size 256 --epochs 50  --fasta-fp  /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/data/4932/4932.protein.sequences.v11.5.fasta --ppi-fp /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/data/4932/4932.protein.physical.links.v11.5.txt --modelname two --learningrate 1e-4 --rawdata_flage
