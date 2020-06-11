import spacy
#python3 -m spacy download en_core_web_lg
import en_core_web_lg
import math

def initialize_hit_list(word_set):
  hit_list = dict();
  for word in word_set:
    hit_list[word] = 0;
  return hit_list;


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
      if(similarity>=0.4):
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
  #print(word_similarity_dict)
  for i in range(0,len(query_tokens)):
    token = query_tokens[i]
    token = (str)(token)
    #print(token)
    #print(word_similarity_dict.keys())
    for key in word_similarity_dict.keys():
      #key = (str)(key)  
      #print(key)  
      if(token == str(key)):
        for similar_word in word_similarity_dict[key]:
            similar_word = (str)(similar_word)
            #print(hit_list[similar_word])
            maximum = hit_list[token]
            if(hit_list[similar_word]>maximum):
                #print("abc",similar_word,token)
                query_tokens[i] = similar_word
                maximum = hit_list[similar_word]
        hit_list[token] = hit_list[token]+1
  return query_tokens    
    #print("gggg",token,hit_list)
  #print(query_tokens)
