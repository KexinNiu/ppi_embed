#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J predict_1_2f
#SBATCH -o predict_1_2f.%J.out
#SBATCH -e predict_1_2f.%J.err
#SBATCH --mail-user=kexin.niu@kaust.edu.sa
#SBATCH --mail-type=end
#SBATCH --time=00:25:00
#SBATCH --mem=20G

printf 'Predict and sv new embeddings for proteins...\n'


# python traindotfeature_0206multi_metric.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --batch-size 64 --epochs 10 --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelname onef -lr 1e-3 --metric dot
# printf '\n_____@@@@@@@@@onef@@@@@@@@@________\n'

python predict_onefdot.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_50_onef_dot_64.th --svpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/pred_result


# printf '\n_____@@@@@@@@@twof@@@@@@@@@@________\n'

# python predict_twofdot.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/  --modelpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_30_twof_dot_64.th --svpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/pred_result

# printf '\n_____@@@@@@@@@onef@@@@@@@@@________\n'
# python predict_onefdot.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/ --modelpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_withabs10_onef_dot_64.th --svpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/pred_result


# printf '\n_____@@@@@@@@@twof@@@@@@@@@@________\n'

# python predict_twofdot.py --data-root /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata --name 4932.physical --esmfolder /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/  --modelpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_withabs30_twof_dot_64.th --svpath /ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/pred_result
