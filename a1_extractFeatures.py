import numpy as np
import pandas as pd
import sys
import argparse
import os
import json
import re
import string

import time

course_dir = '/u/cs401/'
#course_dir = '../'

with open(course_dir + 'Wordlists/First-person') as fp:
    firPer_list = '|'.join(word.lower() for word in fp.read().split())
with open(course_dir + 'Wordlists/Second-person') as fp:
    secPer_list = '|'.join(word.lower() for word in fp.read().split())
with open(course_dir + 'Wordlists/Third-person') as fp:
    thiPer_list = '|'.join(word.lower() for word in fp.read().split())

with open(course_dir + 'Wordlists/Slang') as fp:
    slang_list = '|'.join(word.lower() for word in fp.read().split())

df_BG = pd.read_csv(course_dir + 'Wordlists/BristolNorms+GilhoolyLogie.csv')
df_RW = pd.read_csv(course_dir + 'Wordlists/Ratings_Warriner_et_al.csv')


def extract1(comment):
    '''This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    features = np.zeros(173)

    comment = comment.lower()

    features[0] = len(re.compile(r'(' + firPer_list + r')/prp[$]?').findall(comment))
    features[1] = len(re.compile(r'(' + secPer_list + r')/prp[$]?').findall(comment))
    features[2] = len(re.compile(r'(' + thiPer_list + r')/prp[$]?').findall(comment))
    features[3] = len(re.compile(r'/cc').findall(comment))
    features[4] = len(re.compile(r'/vbd').findall(comment))
    features[5] = len(re.compile(r"('ll|will|gonna|going\S+\sto)/[a-z]+\s[a-z]+/vb").findall(comment))
    features[6] = len(re.compile(r'\/,').findall(comment))
    features[7] = len(re.compile(r'((\/nfp\n)|( ['+string.punctuation+']\/\.\n){2,})').findall(comment))
    features[8] = len(re.compile(r'\/(nn\s)|(nns)').findall(comment))
    features[9] = len(re.compile(r'\/nnps?').findall(comment))
    features[10] = len(re.compile(r'\/rb').findall(comment))
    features[11] = len(re.compile(r'\/(wdt|wp|wrb)').findall(comment))
    features[12] = len(re.compile(slang_list).findall(comment))
    features[13] = len(re.compile(r'[A-Z]+').findall(comment))
    features[14] = len(re.compile(r'\/').findall(comment)) / len(re.compile(r'\n').findall(comment))
    features[15] = np.mean([len(x.split('/')[0]) for x in comment.split() if not re.match(r'.*(\/nfp|\/\.).*', x)])
    features[16] = len(re.compile(r'\n').findall(comment))

    word_list = [token.split('/')[0] for token in comment.split()]

    bg_series = df_BG[df_BG['WORD'].isin(word_list)]
    AoA_array = bg_series[u'AoA (100-700)'].values
    IMG_array = bg_series[u'IMG'].values
    FAM_array = bg_series[u'FAM'].values

    rw_series = df_RW[df_RW['Word'].isin(word_list)]
    VMS_array = rw_series[u'V.Mean.Sum'].values
    AMS_array = rw_series[u'A.Mean.Sum'].values
    DMS_array = rw_series[u'D.Mean.Sum'].values

    features[17] = np.mean(AoA_array)
    features[18] = np.mean(IMG_array)
    features[19] = np.mean(FAM_array)
    features[20] = np.std(AoA_array)
    features[21] = np.std(IMG_array)
    features[22] = np.std(FAM_array)
    features[23] = np.mean(VMS_array)
    features[24] = np.mean(AMS_array)
    features[25] = np.mean(DMS_array)
    features[26] = np.std(VMS_array)
    features[27] = np.std(AMS_array)
    features[28] = np.std(DMS_array)

    return np.nan_to_num(features)


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # TODO: your code here
    cat_to_val = {
        "Right": 2,
        "Left": 0,
        "Center": 1,
        "Alt": 3
    }
    ids = {}
    npys = {}
    for cat in cat_to_val:
        with open(course_dir + 'A1/feats/' + cat + '_IDs.txt') as fp:
            ids[cat] = fp.read().split()
        npys[cat] = np.load(course_dir + 'A1/feats/' + cat + '_feats.dat.npy')

    for index, line in enumerate(data):
        if index % 1000 == 0:
            print(index)
        features = extract1(line['body'])
        features[29:] = npys[line['cat']][ids[line['cat']].index(line['id'])]
        feats[index, :-1] = features
        feats[index, -1] = cat_to_val[line['cat']]

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
