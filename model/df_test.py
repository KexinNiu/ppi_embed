import pandas as pd
import datetime

c6path ='/ibex/scratch/projects/c2014/kexin/ppiproject/esm_atlas/STRING/data/string0.6_cluster.tsv'
# c8path ='/ibex/scratch/projects/c2014/kexin/ppiproject/esm_atlas/STRING/data/string0.8_cluster.tsv'
c8path ='/ibex/scratch/projects/c2014/kexin/ppiproject/esm_atlas/STRING/data/string0.8_clustertest.tsv'
starttime = datetime.datetime.now()
cluster = pd.read_csv(c8path,sep='\t',header=1,names=['class','item'])
loadtime = datetime.datetime.now()
print('loading time =',(loadtime-starttime).seconds)
print(cluster)
# print(cluster.query('class == `item`'))
# cluster.query('1 == `2`')
a = '1347087.CBYO010000006_gene833'
pset = cluster.loc[cluster['item']==a]['item']

# print('pset',type(pset),set(pset))
# pp = pset['item']
# print()


# print(cluster.loc[cluster['class']=='1122940.GCA_000482485_00910'])
# 1122940.GCA_000482485_00926
# print(cluster.loc[cluster['class']=='1122940.GCA_000482485_00926'])

quarytime = datetime.datetime.now()
# print('query time =',(quarytime-loadtime).seconds)
# print('shape,',cluster.shape[0])
# print(set(cluster['class']),len(set(cluster['class'])))
# print('class\n',cluster['class'])

name = ['1122940.GCA_000482485_00926','1187848.AJYQ01000074_gene2117']
name = pd.DataFrame(name,columns=['class'])
print('name=\n',name)

print('1>10 ?',list(cluster[cluster['class'].isin(name['class'])].index))

# adf[adf.x1.isin(bdf.x1)]

print('2>10 ?',cluster['class'][10])