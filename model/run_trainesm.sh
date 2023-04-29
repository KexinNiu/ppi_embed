#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ttrain_esm
#SBATCH -o ttrain_esm.%J.out
#SBATCH -e ttrain_esm.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH --time=00:35:00
#SBATCH --mem=10G

printf 'sim_test,batchsize=8'
python train_esm.py --batch-size 8 --epochs 5 -dr /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/

# /home/niuk0a/.config/code-server
# bind-addr: dgpu212-06:10121
# auth: password
# password: testpass
# cert: false