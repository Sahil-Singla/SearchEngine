import math
import random
import tf_idf
from tf_idf import *

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
      for i in range(0,len(sorted_score_dict)):
        if(len(top_k_docs)<k):
          print(sorted_score_dict[i])
          top_k_docs.append(sorted_score_dict[i][0]);
          #print("cxvcbvnbn",top_k_docs)
  #print(len(top_k_docs))
  #print(top_k_docs)
  return top_k_docs