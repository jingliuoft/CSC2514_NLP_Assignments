from preprocess import *
import pandas as pd
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM

    INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language	: (string) either 'e' (English) or 'f' (French)
    fn_LM		: (string) the location to save the language model once trained

    OUTPUT

    LM          :   (dictionary) a specialized language model

    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which
    incorporate unigram or bigram counts

    e.g., LM['uni']['word'] = 5 # The word 'word' appears 5 times
    LM['bi']['word']['bird'] = 2    # The bigram 'word bird' appears 2 times.
    """

    # TODO: Implement Function
#     def generate_ngrams(s, n):
#         tokens = s.split()
#         ngrams = zip(*[tokens[j:] for j in range(n)])
#         return [" ".join(ngram) for ngram in ngrams]

#     unigrams = []
#     bigrams = []
    sent_prep = ''
    for subdir, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if file == '.DS_Store':
                continue
            fullFile = os.path.join(subdir, file)
            if file[-1] == language:
                read_file = open(fullFile, 'r')
                read_data = read_file.read()
                data = read_data.split('\n')
                for i in range(len(data)):
                    sent_prep += ' '+ preprocess(data[i], file[-1])
    unigrams = generate_ngrams(sent_prep, 1)
    bigrams = generate_ngrams(sent_prep, 2)
    print('preprocess done')
    bigrams_1 = []
    bigrams_2 = []
    df_uni = pd.DataFrame(unigrams, columns=['uni'])
    df_bi = pd.DataFrame(bigrams, columns=['bi'])
    unigrams_u = list(df_uni['uni'].value_counts(sort=False).index)
    unigrams_count = df_uni['uni'].value_counts(sort=False).values
    bigrams_count = df_bi['bi'].value_counts(sort=False).values
    bigrams_u = list(df_bi['bi'].value_counts(sort=False).index)
    for bigram in bigrams_u:
        bigrams_1.append(bigram.split()[0])
        bigrams_2.append(bigram.split()[1])

    LM_uni = dict(zip(unigrams_u, unigrams_count))
    LM_bi = {}
    for x, y, z in zip(bigrams_1, bigrams_2, df_bi['bi'].value_counts(sort=False).values):
        if x in LM_bi:
            LM_bi[x][y] = z
        else:
            LM_bi[x] = {y: z}
    LM = {'uni': LM_uni, 'bi': LM_bi}
    language_model = LM
    print('LM done')


    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return language_model

def generate_ngrams(s, n):
    tokens = s.split()
    #print(tokens)
    ngrams = zip(*[tokens[j:] for j in range(n)])
    return [" ".join(ngram) for ngram in ngrams]