import math
from math import log
import nltk

def sort_dict(dict):
  sorted_list = list();
  for key, value in sorted(dict.items(), key=lambda item: item[1],reverse=True):
    pair = list();
    pair.append(key);
    pair.append(value);
    sorted_list.append(pair);
  #print(sorted_list); 
  return sorted_list;

def get_tokens_from_query(query):
  query_tokens = nltk.word_tokenize(query)
  return query_tokens

def calculate_idf(index,total):
  idf = dict() 
  for key in index:
    idf[key] = log((total/index[key][0]),2);
  return idf;

def get_count_in_query(query_tokens):
  query_dict = dict()
  for word in query_tokens:
    if word in query_dict:
      query_dict[word] = query_dict[word]+1;
    else:
      query_dict[word] = 1; 
  return query_dict;

def query_tf_idf(query_dict,idf):
  denom = 0; 
  query_vec = list()
  for token in query_dict:
    tf = 1+(log(query_dict[token])/log(10))
    if(token in idf.keys()):
      tf_idf_wt = tf*idf[token]
    else:
      tf_idf_wt = 0
    denom = denom+(tf_idf_wt*tf_idf_wt)
    query_vec.append(tf_idf_wt);  
  denom = math.sqrt(denom)  
  for i in range(0,len(query_vec)):
    if(denom!=0):
      query_vec[i] = query_vec[i]/denom;  
  return query_vec

def doc_tf_idf(doc_idx,query_dict,docid):
  doc_vec = list()
  denom = 0;
  for token in doc_idx[docid].keys():
    #print(token)
    denom = denom+(1+log(doc_idx[docid][token])/log(10)); 
  #print("cgvhgjhfg",denom)  
  denom = math.sqrt(denom)
  for token in query_dict:
    if(token in doc_idx[docid].keys()):
      tf = 1+log(doc_idx[docid][token])/log(10)
    else:
      tf = 0  
    if(denom==0):
      print("gffgfhgfg",docid)  
    tf_idf_wt = tf/denom
    doc_vec.append(tf_idf_wt)
  return doc_vec;

def get_lncltc_scores(query_vec,doc_vec):
  score = 0;
  for i in range(0,len(query_vec)):
    score = score+(query_vec[i]*doc_vec[i])
  return score;

def top_k_docs_only(sorted_list,k):
  #print(sorted_list)
  doclist = list();
  for i in range(0,k):
    print(sorted_list[i])  
    doclist.append(sorted_list[i][0]);
  return doclist;

def get_top_k(total_docs,k,query_dict,query_vec,doc_idx):
  score_list = dict()
  for i in range(1,total_docs+1):
    #print(i)
    doc_vec = doc_tf_idf(doc_idx,query_dict,i)
    score = get_lncltc_scores(query_vec,doc_vec)
    score_list[i] = score;
  #print(score_list)
  sorted_scores = sort_dict(score_list);
  #print(sorted_scores)
  top_k_docs = top_k_docs_only(sorted_scores,k);
  return top_k_docs

def process_query(total_docs,k,query_tokens,query_dict,doc_idx,idf):
  #print(query_dict)
  query_vec = query_tf_idf(query_dict,idf)  
  #print(query_vec)
  #print(doc_idx)
  top_k = get_top_k(total_docs,k,query_dict,query_vec,doc_idx)
  return query_vec,top_k;