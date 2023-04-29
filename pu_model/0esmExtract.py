from datapreprocess import extract_esm
import datetime
from pathlib import Path
import pandas as pd


print('ESM features preparation...',datetime.datetime.now(),flush=True)
data_root='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/data/4932'
fasta_fp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/data/4932/4932.protein.sequences.v11.5.fasta'

import argparse

# parser = argparse.ArgumentParser(description='data_root & fastafp')
# parser.add_argument('--data_root')
# parser.add_argument('--fasta_fp')

# args = parser.parse_args()
# data_root = args.data_root
proteins_fp = f'{data_root}/esm/proteins_name2index1.pkl'
esm_fp = f'{data_root}/esm/esm_proteins1.pkl'


# Extract ESM features
esm_dir = Path(f'{data_root}/esm')
proteins, data = extract_esm(fasta_fp, output_dir=esm_dir)
print('ESM features extracted...',datetime.datetime.now(),flush=True)

proteins_name2index = {proteins[i]: i for i in range(len(proteins))}
pd.to_pickle(proteins_name2index,proteins_fp)

proteinsname_fp = proteins_fp.replace('index','name')
pd.to_pickle(proteins,proteinsname_fp)
pd.to_pickle(data,esm_fp)
print('ESM features saved...',datetime.datetime.now(),flush=True)