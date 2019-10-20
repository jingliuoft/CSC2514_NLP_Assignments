from preprocess import *
from lm_train import *
import numpy as np
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
	#TODO: Implement by student.
    sentence = sentence.lower()
    logProb = 0
    if smoothing == False:
        for i in range(len(sentence.split()) - 1):
            if sentence.split()[i] in LM['uni']:
                if sentence.split()[i] in LM['bi']:
                    if sentence.split()[i + 1] in LM['bi'][sentence.split()[i]]:
                        logProb += log(LM['bi'][sentence.split()[i]][sentence.split()[i + 1]] / LM['uni'][sentence.split()[i]], 2)
                    else:
                        logProb += float('-inf')
                else:
                    logProb += float('-inf')
            else:
                logProb += float('-inf')

    else:
        for i in range(len(sentence.split()) - 1):
            if sentence.split()[i] in LM['uni']:
                if sentence.split()[i] in LM['bi']:
                    if sentence.split()[i + 1] in LM['bi'][sentence.split()[i]]:
                        count_ww = LM['bi'][sentence.split()[i]][sentence.split()[i + 1]]
                        count_w = LM['uni'][sentence.split()[i]]
                        logProb += log((count_ww + delta) / (count_w + delta * vocabSize) , 2)
                    else:
                        logProb += log(delta / (LM['uni'][sentence.split()[i]] + delta * vocabSize), 2)
                else:
                    logProb += log(delta / (LM['uni'][sentence.split()[i]] + delta * vocabSize),2)
            else:
                logProb += log(delta / delta * vocabSize, 2)

    return logProb