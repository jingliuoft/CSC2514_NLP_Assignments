#!/usr/bin/python3
# -*- coding: utf-8 -*-

from preprocess import *
from lm_train import *
from log_prob import *
from perplexity import *
from align_ibm1 import *
from decode import *
from BLEU_score import *
from align_ibm2 import *
import parser
import argparse

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    LM = lm_train(data_dir, language, fn_LM)
    return LM
    pass

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    AM = align_ibm2(data_dir, num_sent, max_iter, fn_AM)
    return AM
    pass

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    scores = []
    for i in range(len(eng_decoded)):
        score = BLEU_score(eng_decoded[i], [eng[i],google_refs[i]], n)
        scores.append(score)
    return score
    pass
   

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.

    """
    max_iter = 100
    num_sent = 1000
    data_dir = "../data/Hansard/Training/"
    language = 'e'
    fn_LM = 'e_temp_lm'
    fn_AM = 'e_temp_am'
    bleu_n = 1
    bleu_score = []

    LM = _getLM(data_dir, language, fn_LM)
    AM = _getAM(data_dir, num_sent, max_iter, fn_AM)

    read_file_f = open('../data/Hansard/Testing/Task5.f', 'r')
    read_data_f = read_file_f.read()
    data_f = read_data_f.split('\n')

    read_file_e = open('../data/Hansard/Testing/Task5.e', 'r')
    read_data_e = read_file_e.read()
    data_e = read_data_e.split('\n')
    read_file_er = open('../data/Hansard/Testing/Task5.google.e', 'r')
    read_data_er = read_file_er.read()
    data_er = read_data_er.split('\n')
    #print(len(data_f), len(data_e), len(data_er))

    e=[]
    ref1 = []
    ref2 = []
    for i in range(len(data_f)):
        f_prep = preprocess(data_f[i], 'f')
        e.append(decode(f_prep,LM,AM))
        ref1.append(preprocess(data_e[i], 'e'))
        ref2.append(preprocess(data_er[i], 'e'))

        scores = _get_BLEU_scores(e, ref1, ref2, bleu_n)

    print('done')

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

'''
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(...)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()

    pass
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)