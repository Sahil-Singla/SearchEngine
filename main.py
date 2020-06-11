import json
import index
from index import *
import tf_idf
from tf_idf import *
import improv1
from improv1 import *
import improv2
from improv2 import *
import os
import re
import time
from math import log
from bs4 import BeautifulSoup
import math
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

nltk.download('punkt')

'''
opens file to read given a path and accumulates the doc_list with all docs
'''
def open_file_to_read(path,doc_list):
    #print("1")
    with open(path) as text_file:
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
        #print("2")
        #print(len(html_text.find_all("doc"))) 
        for doc in html_text.find_all("doc"):
            #print(doc)
            filtered = re.sub('[^A-Za-z0-9]+', ' ', doc.text)
            filtered = filtered.lower()
            #print(len(filtered))
            full_text = ""
            for i in range(0,len(filtered)):
                full_text = full_text+filtered[i]
            doc_list.append(full_text)
            #print(filtered)
        #print("3")   
        #print(len(doc_list))

'''
converts json_data in inverted index back to dict
'''
def convert_back_to_dict(path):
    with open(path) as json_file:
        Json_data = json.load(json_file)
        return Json_data

'''
writes relevant documents in the file given by file_path 
'''

def print_docs(doc_list,top_k,file_path):
    for i in range(0,len(top_k)):
        #print(top_k[i])
        f = open(file_path,"w");
        f.write(doc_list[top_k[i]-1])
        f.close()
        

def main():
    doc_list = list();
    invert_idx = dict();
    doc_idx = dict();
    print("Enter the absolute path of the document corpus")
    print("If you do not enter anything, it will take the already present corpus in this folder by default, just press enter to continue")
    file_path = input()
    if(file_path==""):
        file_path = "wiki_011"
    print("File path given is ",file_path)    
    open_file_to_read(file_path,doc_list)
    #use a for loop for docs here
    #list starts with docid 1
    for i in range(0,len(doc_list)):
        invert_idx,doc_idx,word_set = generate_indexes(doc_list[i],i+1,invert_idx,doc_idx);
        #print(doc_idx)
    #print(doc_idx)    
    idx_file = open("invert_index.txt","w");
    idx_file.write(json.dumps(invert_idx))
    idx_file.close()
    print("The output file for the indexes has been generated named invert_index.txt")    
    total_docs = len(doc_list);
    print("Enter the value of K for top K docs between 1 to ",total_docs)
    print("If you do not enter any value, K=min(10,total_docs) by default")
    inp = input()
    if(inp==""):
        k=min(10,total_docs)
    else:    
        k=(int)(inp)
    if(k>total_docs):
        k = total_docs
        print("You have entered a wrong value of k, the total number of docs are ", total_docs)    
        print("Resorting to k = ",total_docs)
    invert_idx_file_path = "invert_index.txt"    
    print("Enter the absolute path of the inverted_index")
    print("If not entered, it will take invert_index.txt by default")
    inp = input()
    if(inp!=""):
        invert_idx_file_path = inp
    invert_idx = convert_back_to_dict(invert_idx_file_path)        
    idf = calculate_idf(invert_idx,total_docs);
    #print(idf)
    hit_list = initialize_hit_list(word_set);
    word_similarity_dict = semantic_similarity(word_set)
    #print(word_similarity_dict)
    leaders = get_leaders(total_docs)
    #print(leaders)
    followers = get_followers(total_docs,leaders)
    #print(followers)
    lf = leader_follower(leaders,followers,doc_idx,idf)
    #print(lf)
    sample_query_file = open("sample.txt", "r")
    for line in sample_query_file:
        query_tokens = get_tokens_from_query(line);
        query_dict = get_count_in_query(query_tokens);
        query_vec,top_k = process_query_semantic_similarity(query_tokens,word_similarity_dict,hit_list);
    #print(idf)
    print("Enter your multi-term query, press 0 to exit")
    query = input()
    while(query!='0'):
        query_tokens = get_tokens_from_query(query);
        query_dict = get_count_in_query(query_tokens);
        print("top_k_documents retreived are ")
        query_vec,top_k = process_query(total_docs,k,query_tokens,query_dict,doc_idx,idf);
        #print(top_k)
        print_docs(doc_list,top_k,"without_improv.txt")
        #print(word_similarity_dict)
        query_tokens = process_query_semantic_similarity(query_tokens,word_similarity_dict,hit_list);
        #print(query_tokens)
        query_dict = get_count_in_query(query_tokens);
        print("top_k_documents retreived with first improvement are ")
        query_vec,top_k = process_query(total_docs,k,query_tokens,query_dict,doc_idx,idf);
        #print(top_k);
        print_docs(doc_list,top_k,"improv1.txt")
        print("top_k_documents retreived with second improvement are ")
        l = get_sorted_leader_list(leaders,doc_idx,query_dict,query_vec)
        top_k = get_top_k_leader_follower(l,lf,k,doc_idx,query_vec,query_dict)
        #print(top_k)
        print_docs(doc_list,top_k,"improv2.txt")
        print("Enter your multi-term query, press 0 to exit")
        query = input()



if __name__ == "__main__":
    main()    