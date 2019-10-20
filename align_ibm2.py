from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import pickle
import os
import copy


def align_ibm2(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm.
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
    AM = {}

    # Read training data
    sents = read_hansard(train_dir, num_sentences)
    print('read hansard done')

    # Initialize AM uniformly
    AM, q = initialize2(sents['en'], sents['fr'])
    print('initialize done')

    # Iterate between E and M steps
    for n in range(max_iter):
        AM, q = em_step_2(AM, q, sents['en'], sents['fr'])

    print('AM done')
    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
    # TODO

    sents_e = []
    sents_f = []
    num_sentences = 1000
    for subdir, dirs, files in os.walk(train_dir):
        for file in sorted(files):
            if file == '.DS_Store':
                continue
            fullFile = os.path.join(subdir, file)
            if file[-1] == 'e':
                read_file_e = open(fullFile, 'r')
                read_data_e = read_file_e.read()
                data_e = read_data_e.split('\n')
                read_file_f = open(fullFile[:-1] + 'f', 'r')
                read_data_f = read_file_f.read()
                data_f = read_data_f.split('\n')
                for i in range(len(data_e)):
                    # print(data)
                    sents_e.append(preprocess(data_e[i], file[-1]))
                    sents_f.append(preprocess(data_f[i], file[-1]))
                    if len(sents_e) == num_sentences:
                        print(num_sentences, ' samples have achived')
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    sents = {
        'en': sents_e,
        'fr': sents_f
    }
    #print(sents['en'])
    return sents


def initialize2(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    # TODO
    unigrams = []
    for i in range(len(eng)):
        unigrams.extend(generate_ngrams(eng[i], 1))

    MA = dict()
    for unigram in unigrams:
        for i in range(len(eng)):
            if unigram in eng[i].split():
                for fw in fre[i].split():
                    if unigram in MA:
                        MA[unigram][fw] = 0
                    else:
                        MA[unigram] = {fw: 0}
    for key in MA:
        for key2 in MA[key]:
            MA[key][key2] = 1 / len(MA[key])

    eng_len = []
    fre_len = []
    for i in range(len(eng)):
        eng_len.append(len(eng[i].split()))
        fre_len.append(len(fre[i].split()))

    q = dict()
    for n in range(len(eng_len)):
        for i in range(eng_len[n]):
            for j in range(fre_len[n]):
                if eng_len[n] in q:
                    if fre_len[n] in q[eng_len[n]]:
                        if i in q[eng_len[n]][fre_len[n]]:
                            q[eng_len[n]][fre_len[n]][i][j] = 0
                        else:
                            q[eng_len[n]][fre_len[n]][i] = {j: 0}
                    else:
                        q[eng_len[n]][fre_len[n]] = {i: {j: 0}}
                else:
                    q[eng_len[n]] = {fre_len[n]: {i: {j: 0}}}
    for key in q:
        for key2 in q[key]:
            for key3 in q[key][key2]:
                for key4 in q[key][key2][key3]:
                    q[key][key2][key3][key4] = 1 / len(q[key][key2][key3])

    return MA, q


def em_step_2(MA, q, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    # TODO

    MA_Tcount = copy.deepcopy(MA)
    for e in MA_Tcount:
        for f in MA_Tcount[e]:
            MA_Tcount[e][f] = 0

    MA_count = dict()
    for key in MA:
        MA_count[key] = 0

    eng_len = []
    fre_len = []
    for i in range(len(eng)):
        eng_len.append(len(eng[i].split()))
        fre_len.append(len(fre[i].split()))

    RC_count = copy.deepcopy(q)
    for key in RC_count:
        for key2 in RC_count[key]:
            for key3 in RC_count[key][key2]:
                for key4 in RC_count[key][key2][key3]:
                    RC_count[key][key2][key3][key4] = 0

    RC_Tcount = dict()
    for n in range(len(eng_len)):
        for i in range(eng_len[n]):
            if eng_len[n] in RC_Tcount:
                if fre_len[n] in RC_Tcount[eng_len[n]]:
                    RC_Tcount[eng_len[n]][fre_len[n]][i] = 0
                else:
                    RC_Tcount[eng_len[n]][fre_len[n]] = {i: 0}
            else:
                RC_Tcount[eng_len[n]] = {fre_len[n]: {i: 0}}

    for l in range(len(eng)):
        eng_s = eng[l].split()
        fre_s = fre[l].split()

        for f in list(set(fre_s)):
            denom_c = 0
            for e in list(set(eng_s)):
                denom_c += (MA[e][f] * fre_s.count(f))
            for e in list(set(eng_s)):
                # print(eng_s[ji],fre_s[i],MA[eng_s[ji]][fre_s[i]])
                MA_Tcount[e][f] += MA[e][f] * fre_s.count(f) * eng_s.count(e) / denom_c
                MA_count[e] += MA[e][f] * fre_s.count(f) * eng_s.count(e) / denom_c
        # print('done {} sententce of length {}, total {}'.format(l, len(eng_s), len(eng)))
        for i in range(len(eng_s)):
            for j in range(len(fre_s)):
                RC_count[len(eng_s)][len(fre_s)][i][j] += MA[eng_s[i]][fre_s[j]] / denom_c
                RC_Tcount[len(eng_s)][len(fre_s)][i] += MA[eng_s[i]][fre_s[j]] / denom_c
    for e in MA:
        for f in MA[e]:
            MA[e][f] = MA_Tcount[e][f] / MA_count[e]
    for key in q:
        for key2 in q[key]:
            for key3 in q[key][key2]:
                for key4 in q[key][key2][key3]:
                    q[key][key2][key3][key4] = RC_count[key][key2][key3][key4] / RC_Tcount[key][key2][key3]

    return MA, q