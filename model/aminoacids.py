import numpy as np

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i + 1
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])
MAXLEN = 2000
NGRAMS = {}
for i in range(20):
    for j in range(20):
        for k in range(20):
            ngram = AALETTER[i] + AALETTER[j] + AALETTER[k]
            index = 400 * i + 20 * j + k + 1
            NGRAMS[ngram] = index

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

def to_ngrams(seq):
    l = min(MAXLEN, len(seq) - 3)
    ngrams = np.zeros((l,), dtype=np.int32)
    for i in range(l):
        ngrams[i] = NGRAMS.get(seq[i: i + 3], 0)
    return ngrams

def to_onehot(seq, start=0):
    onehot = np.zeros((MAXLEN, 21), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot



# pr = 'MAAATTTTTTSSSISFSTKPSPSSSKSPLPISRFSLPFSLNPNKSSSSSRRRGIKSSSPSSISAVLNTTTNVTTTPSPTKPTKPETFISRFAPDQPRKGADILVEALERQGVETVFAYPGGTSMEIHQALTRSSSIRNVLPRHEQGGVFAAEGYARSSGKPGICIATSGPGATNLVSGLADALLDSVPLVAITGQVPRRMIGTDAFQETPIVEVTRSITKHNYLVMDVEDIPRIIEEAFFLATSGRPGPVLVDVPKDIQQQLAIPNWEQAMRLPGYMSRMPKPPEDSHLEQIVRLISESKKPVLYVGGGCLNSSDELGRFVELTGIPVASTLMGLGSYPCDDELSLHMLGMHGTVYANYAVEHSDLLLAFGVRFDDRVTGKLEAFASRAKIVHIDIDSAEIGKNKTPHVSVCGDVKLALQGMNKVLENRAEELKLDFGVWRNELNVQKQKFPLSFKTFGEAIPPQYAIKVLDELTDGKAIISTGVGQHQMWAAQFYNYKKPRQWLSSGGLGAMGFGLPAAIGASVANPDAIVVDIDGDGSFIMNVQELATIRVENLPVKVLLLNNQHLGMVMQWEDRFYKANRAHTFLGDPAQEDEIFPNMLLFAAACGIPAARVTKKADLREAIQTMLDTPGPYLLDVICPHQEHVLPMIPSGGTFNDVITEGDGRIKY'
# # one = to_onehot(pr)
# # print(len(to_ngrams(one)))

# pr = 'MAAATTTTTTSSSISFSTKPSPSSSKSPLPISRFSLPFSLNPY'

# # # print('onhot:')
# # print(len(to_onehot(pr)))
# # # print('to_ngrams:')
# print(len(to_ngrams(pr)))
'''
onhot:
[[0 0 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 ...
 [1 0 0 ... 0 0 0]
 [1 0 0 ... 0 0 0]
 [1 0 0 ... 0 0 0]]
to_ngrams:
[4801    1   17  337 6737 6737 6737 6737 6736 6716 6316 6310 6196 3914
 6276 5517 6332 6635 4696 5915 6296 5916 6316 6312 6236 4715 6291 5815
 4290 5796 3902 6034  676 5511 6215 4294 5876 5511 6203 4055 1083 5652
 1036 4716 6316 6316 6316 6302 6022  422  428  550 2992 3836 4716 6316
 6315 6296 5916 6310 6196 3901 6020  391 7803 4057 1137 6737 6723 6460
 1197 7937 6737 6735 6696 5915 6297 5932 6635 4697 5932 6635 4687 5737
 2734 6670 5396 3902 6034  661 5215  284 5666 1315 2282 5632  628 4541
 2804   70 1391 3820 4387 7721 2411  207 4122 2426  508 2160 3187 7737
 2740 6794 7861 5219  375 7488 5748 2957 3136 6713 6247 4930 2589 3766
 3301 2011  217 4322 6436  716 6316 6310 6182 3623  460 1191 7815 4282
 5629  567 3326 2508 2148 2960 3194 7861 5201    7  128 2559 3161 7202
   36  716 6308 6152 3035 4688 5750 2985 3690 1781 3617  336 6708 6155
 3088 5741 2817  323 6451 1020 4396 7908 6151 3001 4004   61 1211  211
 4204 4076 1520 6395 7891 5820 4381 7610  197 3928 6546 2920 2395 7882
 5622  433  650 4988 3757 3124 6461 1214  266 5307 2137 2735 6690 5800
 3987 7740 2797 7922 6436  710 6197 3932 6629 4563 3259 1171 7420 4393
 7844 4880 1587 7724 2470 1395 3882 5630  590 3787 3727 2521 2414  274
 5471 5401 4017  336 6708 6142 2835  688 5755 3100 5991 7820 4384 7680
 1595 7892 5824 4470 1386 3706 2106 2111 2201 4010  195 3883 5658 1147
 6926 2501 2013  242 4831  615 4288 5759 3173 7456 5102 6033  655 5092
 5835 4695 5887 5724 2476 1509 6171 3407
'''
# import torch
# import torch.nn as nn

# input1 = torch.randn(100, 128)
# input2 = torch.randn(100, 128)
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(input1, input2)
# print(output)

# pdist = nn.PairwiseDistance(p=2)
# output = pdist(input1, input2)
# print(output)

# cosemb = nn.CosineEmbeddingLoss(margin =1)
# label = torch.tensor([-1])
# label = label.repeat(4)
# print('label',label)
# x= x.repeat(4, 2)
# input11 = input1.repeat(4,0)
# print('input11',input11)
# input11 = input1.repeat(4,1)
# print('input11',input11,input11.size())
# input11 = torch.cat((input1, input1, input1), 1)
# print('input11',input11,input11.size())

# input22 = input2.repeat(4,2)

# print('input1',input1.size(),input1)

# print('input11',input11.size())
# input11 = torch.tensor([input1,input1,input1,input1])
# input22 = torch.tensor([input2,input2,input2,input2])
# output = cosemb(input11,input22,label)
# print(output)
