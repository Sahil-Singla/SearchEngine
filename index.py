import nltk
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.stem import PorterStemmer    
from nltk.stem import WordNetLemmatizer

'''
generates all relevant indexes
'''

def generate_indexes(text,docid,invert_idx,doc_idx):
  word_set = set()
  tokens = nltk.word_tokenize(text);
  #print(tokens)
  doc_idx[docid] = dict();
  for key in tokens:
    key = (str)(key)
    word_set.add(key);
    if(key in doc_idx[docid]):
      doc_idx[docid][key] = doc_idx[docid][key]+1;
    else:
      doc_idx[docid][key] = 1;  
    if(key in invert_idx.keys()):
      if(len(invert_idx[key])>1 and invert_idx[key][-1][0]!=docid):
        invert_idx[key][0] = invert_idx[key][0]+1;
        freq_docid = list();
        freq_docid.append(1);
        freq_docid.append(docid);
        invert_idx[key].append(freq_docid);
      else:
        invert_idx[key][-1][1] = invert_idx[key][-1][1]+1;  
    else:
      invert_idx[key] = list();
      invert_idx[key].append(1);
      freq_docid = list();
      freq_docid.append(docid);
      freq_docid.append(1);
      invert_idx[key].append(freq_docid);
  #print(len(invert_idx))    
  #print(invert_idx)
  return invert_idx,doc_idx,word_set

  