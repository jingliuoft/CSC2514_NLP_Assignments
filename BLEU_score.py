import math
import numpy as np
from lm_train import *


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

    # TODO: Implement by student.
    p = 1
    for i in range(n):
        n_gram = i+1
        cand_gram = generate_ngrams(candidate, n_gram)
        ref_gram = []
        r = []
        for j in range(len(references)):
            ref_gram.extend(generate_ngrams(references[j], n_gram))
            r.append(len(references[j].split()))
        count = 0
        for gram in cand_gram:
            if gram in ref_gram:
                count += 1
        p *= (count/len(cand_gram))
    if brevity == True:
        c = len(candidate.split())
        min = c
        for ri in r:
            if np.abs(ri-c) <= min:
                min = ri
        if ri/c <1:
            BP = 1
        else:
            BP = np.exp(1-ri/c)
        bleu_score = BP*p
    else:
        bleu_score = p

    return bleu_score
