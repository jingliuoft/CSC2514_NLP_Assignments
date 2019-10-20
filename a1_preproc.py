import sys
import argparse
import os
import json
import unicodedata
import re
import string
import spacy

course_dir = '/u/cs401/'
#course_dir = '../'

abbrev_list = []
with open(course_dir + 'Wordlists/abbrev.english') as fp:
    abbrev_list = abbrev_list + fp.read().split()
with open(course_dir + 'Wordlists/pn_abbrev.english') as fp:
    abbrev_list = abbrev_list + fp.read().split()
abbrev_list = [abbrev.lower() for abbrev in abbrev_list]

stopWords = []
with open(course_dir + 'Wordlists/StopWords') as fp:
    stopWords = fp.read().split()

nlp = spacy.load('en', disable=['parser', 'ner'])


def preproc1(comment, steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:
        modComm = re.sub(r'[\n]+', ' ', modComm)

    if 2 in steps:
        modComm = unicodedata.normalize('NFKD', modComm).encode('ascii', 'ignore').decode("ascii")

    if 3 in steps:
        # limitation: will capture improper use of '.', for example, hello.world will be recognized as a hyperlink.
        # to avoid the limitation, additional packages need to be used: https://pypi.org/project/tld/ to identify
        # valid top level domain names such as .com, .org, etc.
        modComm = re.sub(r'(http[s]?:\/\/)?((([a-zA-Z0-9]{1,}\.)+[a-zA-Z]{2,5})|(([0-9]+\.){3}[0-9]+))([\S]*)?[^\.\)\'\s]', '', modComm)

    if 4 in steps:
        modCommSplit = modComm.split()

        index = 0
        while index < len(modCommSplit):
            word = modCommSplit[index]
            # for apostrophes and periods in abbreviations, add the matched part to modCommSplit and the rest back to
            # modCommSplit to process in next iteration
            # Apostrophes
            if re.search(r".*['].*", word) is not None:
                temp_word_sep = re.sub(r"(.*['][a-zA-Z]*)(.*)", r'\1 \2', word).split()
                modCommSplit[index] = temp_word_sep[0]
                if len(temp_word_sep) > 1 and len(temp_word_sep[1]) > 0:
                    modCommSplit.insert(index+1, temp_word_sep[1])
                index = index + 1
                continue
            # Periods in abbreviations
            found = False
            for abbrev in abbrev_list:
                if word.lower().find(abbrev) == 0:
                    modCommSplit[index] = abbrev
                    rest_word = word[len(abbrev):]
                    if len(rest_word) > 0:
                        modCommSplit.insert(index+1, rest_word)
                    found = True
                    break
            if found:
                index = index + 1
                continue

            # split punctuations, including single hyphens
            # for non single hyphen case, the \3 part will create extra space at the end, therefore the strip() is used
            modCommSplit[index] = re.sub(r'([a-zA-Z0-9]*)([' + string.punctuation + ']+)([a-zA-Z0-9]*)', r'\1 \2 \3', word).strip()
            index = index + 1
            continue

        modComm = " ".join(modCommSplit)

    if 5 in steps:
        modComm = re.sub(r"can't", "can n't", modComm)
        modComm = re.sub(r"([a-zA-Z])(n't)", r'\1 \2', modComm)  # to avoid  extra space for can't
        modComm = re.sub(r"'s", " 's", modComm)
        modComm = re.sub(r"s'", "s '", modComm)
        modComm = re.sub(r"'re", " 're", modComm)
        modComm = re.sub(r"'m", " 'm", modComm)
        modComm = re.sub(r"'ve", " 've", modComm)
        modComm = re.sub(r"'ll", " 'll", modComm)
        modComm = re.sub(r"'d", " 'd", modComm)

    if 6 in steps or 7 in steps or 8 in steps:
        modComm = " ".join(word for word in modComm.split() if word not in stopWords)
        utt = nlp(modComm)
        tagged_tokens = [(token.text if token.lemma_[0] == '-' else token.lemma_) + '/' + token.tag_ for token in utt]
        modComm = " ".join(tagged_tokens)

    if 9 in steps:
        modComm = re.sub(r'/\.', '/.\n', modComm)
        modComm = re.sub(r'/NFP', '/NFP\n', modComm)
        if modComm[-1:] != '\n':
            modComm = modComm + '\n'

    if 10 in steps:
        modComm = modComm.lower()

    return modComm


def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(course_dir+'A1/data'):
        for file in files:
            if file =='.DS_Store':
                continue
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            print(len(data))

            # select appropriate args.max lines
            startingIndex = args.ID[0] % len(data)
            for i in range(10000):
                line = data[(startingIndex+i) % len(data)]
                # read those lines with something like `j = json.loads(line)`
                j = json.loads(line)
                # choose to retain fields from those lines that are relevant to you
                newJ = {'id': j['id']}
                # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                newJ['cat'] = file
                # process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # replace the 'body' field with the processed text
                newJ['body'] = preproc1(j['body'])

                # append the result to 'allOutput'
                allOutput.append(newJ)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)