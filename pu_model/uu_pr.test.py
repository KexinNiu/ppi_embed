# pd.options.mode.chained_assignment = None
#     '''
#     predresult = pd.DataFrame({
#                 "p1":p1,
#                 "p2":p2,
#                 "labels": alllabel,
#                 "preds":preds
#                 })
#     '''

#     ranks={} ## key = protein, val
#     uniprotiein = np.unique(predresult[["p1", "p2"]].values)
#     n = len(uniprotiein)
#     cn = n
#     print('nnnn ={}'.format(n),flush=True)

def computeAVP(predresult,protein):
    # https://github.com/DavidRFerreira/InformationRetrieval_EvaluationMetrics/blob/main/src/computeMetrics.py
    pd.options.mode.chained_assignment = None
    precisionlist=[]
    recalllist=[]
    totalRelevantAlreadyFound = 0
    protein_preds = predresult.query("p1==@protein")
    rkll = rankdata(protein_preds['preds'].values, method='average')
    protein_preds[ 'rankval'] = rkll

    sort_preds = protein_preds.sort_values(by=['rankval'])
    pos_rankvals = protein_preds.query('labels==1.0')['rankval']
    numTotalRelevant = len(pos_rankvals)

    for i in range(0,len(sort_preds)):
        if sort_preds.at[i,'labels'] ==1:
            totalRelevantAlreadyFound += 1
            averagePrecision = averagePrecision + totalRelevantAlreadyFound / (i + 1)    
        precisionlist.append(totalRelevantAlreadyFound/(i+1))
        recalllist.append(totalRelevantAlreadyFound/numTotalRelevant)
    
    averagePrecision = averagePrecision/ totalRelevantAlreadyFound
    print("\nSET:")
    print(set)
    print("\nPRECISION:")
    print(precisionlist)
    print("\nRECALL:")
    print(recalllist)
    print("\nAVERAGE PRECISION:")
    print(averagePrecision)
    print("")
    return precisionlist,recalllist,averagePrecision,numTotalRelevant

def compute_MAP(predresult):
    uniprotiein = np.unique(predresult[["p1", "p2"]].values)
    n = len(uniprotiein)
    precisionSet=[]
    recallSet = []
    averagePrecision =0
    avpSet=[]
    for protein in uniprotiein:
        precision,recall,averagePrecision,numTotalRelevant = computeAVP(predresult,protein)
        avpSet.append(averagePrecision)
        precisionSet.append(precision)
        recallSet.append(recall)
    MAP = statistics.mean(avpSet)
    print("\nMAP:")
    print(MAP)