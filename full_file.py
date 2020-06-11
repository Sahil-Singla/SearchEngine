import os
import re
import time
from math import log
from bs4 import BeautifulSoup
import math
import pandas as pd
import random
from matplotlib import pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.stem import PorterStemmer    
from nltk.stem import WordNetLemmatizer
!python -m spacy download en
!python -m spacy download en_core_web_lg
import spacy
import en_core_web_lg

nltk.download('punkt')

def sort_dict(dict):
  sorted_list = list();
  for key, value in sorted(dict.items(), key=lambda item: item[1],reverse=True):
    pair = list();
    pair.append(key);
    pair.append(value);
    sorted_list.append(pair);
  #print(sorted_list); 
  return sorted_list

def generate_indexes(text,docid,invert_idx,doc_idx):
  word_set = set()
  tokens = nltk.word_tokenize(text);
  #print(tokens)
  doc_idx[docid] = dict();
  for key in tokens:
    word_set.add(key);
    if(key in doc_idx[docid]):
      doc_idx[docid][key] = doc_idx[docid][key]+1;
    else:
      doc_idx[docid][key] = 1;  
    if(key in invert_idx.keys()):
      if(len(invert_idx[key])>1 and invert_idx[key][-1][1]!=docid):
        invert_idx[key][0] = invert_idx[key][0]+1;
        freq_docid = list();
        freq_docid.append(1);
        freq_docid.append(docid);
        invert_idx[key].append(freq_docid);
      else:
        invert_idx[key][-1][0] = invert_idx[key][-1][0]+1;  
    else:
      invert_idx[key] = list();
      invert_idx[key].append(1);
      freq_docid = list();
      freq_docid.append(1);
      freq_docid.append(docid);
      invert_idx[key].append(freq_docid);
  #print(len(invert_idx))    
  #print(invert_idx)
  return invert_idx,doc_idx,word_set

def get_tokens_from_query(query):
  query_tokens = nltk.word_tokenize(query)
  query_dict = get_count_in_query(query_tokens);
  return query_tokens,query_dict

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
    doclist.append(sorted_list[i][0]);
  return doclist;

def get_top_k(total_docs,k,query_dict,query_vec):
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

def process_query(query):
  query_tokens,query_dict = get_tokens_from_query(query);
  #print(query_dict)
  query_vec = query_tf_idf(query_dict,idf)  
  #print(query_vec)
  #print(doc_idx)
  top_k = get_top_k(total_docs,k,query_dict,query_vec)
  return query_tokens,query_dict,query_vec,top_k;

def initialize_hit_list(word_set):
  hit_list = dict();
  for word in word_set:
    hit_list[word] = 0;
  return hit_list;

print(semantic_similarity(word_set))

'''
semantic similarity using spacy + concept of hits using multiple queries
if there is a similar word for a token in the query having a higher search rate, we replace the word in the query by
that word, can provide much more relevant information to the user.
'''

def semantic_similarity(word_set):
  similar = en_core_web_lg.load(); 
  word_vec = list();
  for word in word_set:
    word_vec.append(word);
  word_similarity = dict();
  for i in range(0,len(word_vec)):
    word_vec[i] = similar(word_vec[i]);
  #print(word_vec)
  for i in range(0,len(word_vec)):
    for j in range(0,i):
      #word_vec[i] = (str)(word_vec[i]);
      #word_vec[j] = (str)(word_vec[j]);
      similarity = word_vec[i].similarity(word_vec[j]);
      #print(word_vec[i],word_vec[j],similarity)
      if(similarity>=0.8):
        if(word_vec[i] in word_similarity):
          word_similarity[word_vec[i]].append(word_vec[j]);
        else:
          word_similarity[word_vec[i]] = list();
          word_similarity[word_vec[i]].append(word_vec[j]);
        if(word_vec[j] in word_similarity):
          word_similarity[word_vec[j]].append(word_vec[i]);
        else:
          word_similarity[word_vec[j]] = list();
          word_similarity[word_vec[j]].append(word_vec[i]);        
  return word_similarity

def process_query_semantic_similarity(query_tokens,word_similarity_dict,hit_list):
  for i in range(0,len(query_tokens)):
    token = query_tokens[i]
    token = (str)(token)
    if(token in word_similarity_dict.keys()):
      for similar_word in word_similarity_dict[token]:
        similar_word = (str)(similar_word)
        print(hit_list[similar_word])
        if(hit_list[similar_word]>hit_list[token]):
          #print("abc",similar_word,token)
          query_tokens[i] = similar_word;
      hit_list[token] = hit_list[token]+1;
    #print("gggg",token,hit_list)
  #print(query_tokens)

def leader_follower(leaders,followers,doc_idx,idf):
  leader_follower_relation = dict();
  for leader in leaders:
    leader_follower_relation[leader] = list()
  for follower in followers:
    follower_vec = query_tf_idf(doc_idx[follower],idf);
    leader_scores = dict();
    for leader in leaders:
      leader_vec = doc_tf_idf(doc_idx,doc_idx[follower],leader);
      score = get_lncltc_scores(follower_vec,leader_vec);
      leader_scores[leader] = score;
    sorted_leaders = sort_dict(leader_scores);
    top = sorted_leaders[0][0];
    #print(top)
    if(top in leader_follower_relation):
      leader_follower_relation[top].append(follower);
  return leader_follower_relation;

#improvement1 -> Leader-follower-model
def get_leaders(total_docs):
  leaders = set();
  root = math.sqrt(total_docs);
  while(len(leaders)<root):
    random_number = random.randint(1,total_docs);
    leaders.add(random_number);
    #print("vvvggggjj",random_number)
  return leaders

def get_sorted_leader_list(leaders,doc_idx,query_dict,query_vec):
  leader_scores = dict();
  for i in leaders:
    doc_vec = doc_tf_idf(doc_idx,query_dict,i);
    score = get_lncltc_scores(query_vec,doc_vec);
    if(not(i in leader_scores)):
        leader_scores[i] = score;
  sorted_leaders = sort_dict(leader_scores)      
  return sorted_leaders

def get_followers(total_docs,leaders):
  followers = set();
  for i in range(1,total_docs+1):
    if(not(i in leaders)):
      followers.add(i);
  return followers;

def get_top_k_leader_follower(sorted_leaders,leader_follower_dict,k,doc_idx,query_vec,query_dict):
  top_k_docs = list();
  #print(k)
  while(len(top_k_docs)<k):
    for leader in sorted_leaders:
      leader_doc_id = leader[0]
      #print(leader_doc_id)
      score_dict = dict()
      leader_vec = doc_tf_idf(doc_idx,query_dict,leader_doc_id)
      score_dict[leader_doc_id] = get_lncltc_scores(query_vec,leader_vec);
      #print("sdgdffgh",leader_follower_dict)
      #print(leader_doc_id)
      for follower in leader_follower_dict[leader_doc_id]:
        follower_doc_vec = doc_tf_idf(doc_idx,query_dict,follower)
        score_dict[follower] = get_lncltc_scores(query_vec,follower_doc_vec)
      sorted_score_dict = sort_dict(score_dict)
      #print("qewqwt",sorted_score_dict)  
      for i in range(0,len(sorted_score_dict)):
        if(len(top_k_docs)<k):
          top_k_docs.append(sorted_score_dict[i][0]);
          #print("cxvcbvnbn",top_k_docs)
  #print(len(top_k_docs))
  #print(top_k_docs)
  return top_k_docs

#open the file to read, worked on AK/wiki_01
with open('wiki_011') as text_file:
  html_text = BeautifulSoup(text_file,'html.parser')
#print(html_text)

tags = ['a']
#simple function to get text inside <a></a> tags. Example: <a href="www.google.com"> Google </a> gives Google.
def getString(s):
    a_str = s.string
    if(a_str):
        return a_str
    else:
        return ""
        
for t in tags:
    #print(html(t))
    sentences = html_text(t)
    if(t=='a'):
        for s in sentences:
         extracted = getString(s)
         s.replace_with(extracted)  

doc_list = list()
for doc in html_text.find_all("doc"):
    #print(doc)
    full_text = ""
    for i in range(0,len(doc.text)):
      full_text = full_text+doc.text[i]
    #print(len(full_text))
    filtered = re.sub('[^A-Za-z0-9]+', ' ', doc.text)
    filtered = filtered.lower()
    doc_list.append(filtered)

invert_idx = dict();
doc_idx = dict();
#use a for loop for docs here
#list starts with docid 1
for i in range(0,len(doc_list)):
  invert_idx,doc_idx,word_set = generate_indexes(doc_list[i],i+1,invert_idx,doc_idx);
#print(doc_idx)
total_docs = len(doc_list);
k=10
idf = calculate_idf(invert_idx,total_docs);
hit_list = initialize_hit_list(word_set);
word_similarity_dict = semantic_similarity(word_set)

leaders = get_leaders(total_docs)
#print(leaders)
followers = get_followers(total_docs,leaders)
#print(followers)
lf = leader_follower(leaders,followers,doc_idx,idf)
#print(lf)

#print(idf)

query1 = "you mary"
query2 = "know mary"

query_tokens,query_dict,query_vec,top_k = process_query(query1);


print(top_k)
#print(word_similarity_dict)
process_query_semantic_similarity(query_tokens,word_similarity_dict,hit_list);
#print(query_tokens)

query_tokens,query_dict,query_vec,top_k = process_query(query2);
process_query_semantic_similarity(query_tokens,word_similarity_dict,hit_list);
print(top_k)
#print(query_tokens)

l = get_sorted_leader_list(leaders,doc_idx,query_dict,query_vec)
#print(l)
top_k = get_top_k_leader_follower(l,lf,k,doc_idx,query_vec,query_dict)
print(top_k)