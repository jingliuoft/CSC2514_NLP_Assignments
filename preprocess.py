import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function

    in_sentence = in_sentence.lower()
    out_sentence = re.sub(r'([a-zA-Z0-9]*)([!"#\$%&\(\)*\+,\.\/:;<=>?@\[\\\]\^_`{\|}~])([a-zA-Z0-9]*)',
                         r'\1 \2 \3', in_sentence)
    out_sentence = re.sub(r'([0-9])(-)([0-9])', r'\1 \2 \3', out_sentence)
    out_sentence = re.sub(r'(\()([\s|\S]*)(-)([\s|\S]*)(\))', r'\1\2 \3 \4\5', out_sentence)
    out_sentence = ' '.join(out_sentence.split())
    if language == 'f':
        out_sentence = re.sub(r'(^[a-z])(\')', r'\1\2 ', out_sentence)
        out_sentence = re.sub(r'(\s[a-z])(\')', r'\1\2 ', out_sentence)
        out_sentence = re.sub(r'(d\')\s(abord|accord|ailleurs|habitude)', r'\1\2', out_sentence)
        out_sentence = re.sub(r'^qu\'', '^qu\' ', out_sentence)
        out_sentence = re.sub(r'\squ\'', '\squ\' ', out_sentence)
        out_sentence = re.sub(r'^puisqu\'', '^puisqu\' ', out_sentence)
        out_sentence = re.sub(r'\spuisqu\'', '\spuisqu\' ', out_sentence)
        out_sentence = re.sub(r'^lorsqu\'', '^lorsqu\' ', out_sentence)
        out_sentence = re.sub(r'\slorsqu\'', '\slorsqu\' ', out_sentence)

    return 'sentstart '+ out_sentence + ' sentend'

