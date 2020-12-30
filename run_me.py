#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 00:02:38 2020

@author: yasmin
"""

import numpy as np
from sklearn import metrics
from sklearn.neighbors import BallTree
from pathlib import Path
import os
import datetime
import random

num_dimensions = 29
num_folds_to_test = 5

#####################################################
#####################################################
#KNN specific functions

#function to return balltree using sklearn module for knn training
def get_tree(rd_lst): 
    rd_lst = np.array(rd_lst) #converting training lst to form suitable for BallTree func
    tree = BallTree(rd_lst)
    
    return tree

#function to cast majority vote for knn model
def cast_majority_vote_knn(neighbors_label_idx_lst, label_lst):
    normal_cnt = 0
    fraud_cnt = 0

    for idx in neighbors_label_idx_lst:
        if(label_lst[idx] == "Normal"):
            normal_cnt += 1
        else:
            fraud_cnt += 1
            
    if(fraud_cnt > normal_cnt):
        return "Fraud"
    else:
        return "Normal"

#function to eval knn tree provided by querying predicted label for each point in tst list
def eval_knn_tree(num_neighbors, tr_lst, tst_lst, tr_label_lst):
    tr_tree = get_tree(tr_lst)  
    y_lst = []
    
    for i in range(len(tst_lst)):
        dist, ind = tr_tree.query(tst_lst[i:i+1], k=num_neighbors)
        label_vote = cast_majority_vote_knn(ind[0], tr_label_lst)
        y_lst.append(label_vote)
    
    return y_lst

#####################################################
#####################################################
#Decision Tree specific functions

#class for decision tree node definition
class dec_tree_node:
    def __init__(self, depth, dim_info):
        self.depth = depth
        self.dim_idx = dim_info[0]
        self.dim_split_val = dim_info[1]
        self.left_child = None
        self.right_child = None
        self.has_l_leaf = 0
        self.has_r_leaf = 0
        self.l_label_vote = "None"
        self.r_label_vote = "None"
    
    def print_node_info(self):
        print("Node depth is", self.depth, "best dim is", self.dim_idx, "and best split val for this dim is", self.dim_split_val, "has_leaf val is", self.has_l_leaf, self.has_r_leaf, "label is", self.l_label_vote, self.r_label_vote)
    
    def add_l_child(self, child):
        self.left_child = child
        
    def add_r_child(self, child):
        self.right_child = child
        
    def add_l_leaf(self, label):
        self.has_l_leaf = 1
        self.l_label_vote = label
        
    def add_r_leaf(self, label):
        self.has_r_leaf = 1
        self.r_label_vote = label
        
#print tree info      
def print_tree(node):
    if(node.has_r_leaf or node.has_l_leaf):
        node.print_node_info()
        return
    
    if(not(node.left_child == None)):
        print_tree(node.left_child)
        
    if(not(node.right_child == None)):
        print_tree(node.right_child)    

#function to split dataset in two chunks using given dimension info        
def split_data(tr_data, labels, dim_info):
    dim_idx = dim_info[0] #best dimension index
    split_val = dim_info[1] #best split val for given dimension index
    left_dat = []
    right_dat = []
    left_labels = []
    right_labels = []
    
    for pt_idx in range(len(tr_data)):
        pt = tr_data[pt_idx]
        if (pt[dim_idx] < split_val):
            left_dat.append(pt)
            left_labels.append(labels[pt_idx])
        else:
            right_dat.append(pt)
            right_labels.append(labels[pt_idx])            
    #return two datasets, two label lists for left and right data sets                
    return left_dat, right_dat, left_labels, right_labels 

#criterion function for determining which dimension is best for given data/(analagous to gini index)
#takes sum of absolute differences on either side <split val , > split val to determine which split
#val provides the most info
def get_sum_of_differences(dim_val_lst, dim_label_lst, num_splits=10):
    max_val = max(dim_val_lst)
    min_val = min(dim_val_lst)
    split_val_lst = []
    sum_diff_lst = []
    
    dim_val_range = (max_val - min_val)
    split_inc_val = dim_val_range/(num_splits+1) 
    #split val unit to increment by for each split, divides data range over num splits
    
    for i in range(1, num_splits):   
        normal_cnt_lesser_than = 0
        fraud_cnt_lesser_than = 0
        
        normal_cnt_greater_than = 0
        fraud_cnt_greater_than = 0
        split_val = min_val + (i*split_inc_val)
        split_val_lst.append(split_val)

        for pt_idx in range(len(dim_val_lst)):
            pt_val = dim_val_lst[pt_idx]
    
            if(pt_val < split_val):
                if(dim_label_lst[pt_idx] == "Normal"):
                    normal_cnt_lesser_than += 1
                else:
                    fraud_cnt_lesser_than += 1
                    
            elif(pt_val > split_val):
                if(dim_label_lst[pt_idx] == "Normal"):
                    normal_cnt_greater_than += 1
                else:
                    fraud_cnt_greater_than += 1      

#get absolute differences between normal and fraud counts on either side of split, 
#e.g. unanimous normal will give higher score than one with lower winning margin
        less_than_diff = abs(normal_cnt_lesser_than - fraud_cnt_lesser_than)
        greater_than_diff = abs(normal_cnt_greater_than - fraud_cnt_greater_than)
        sum_of_differences = less_than_diff + greater_than_diff #sum the absolute differences
        sum_diff_lst.append(sum_of_differences)
    
    max_sum = max(sum_diff_lst)
    best_split_idx = sum_diff_lst.index(max_sum)
    best_split_val = split_val_lst[best_split_idx]
    
    #return highest score and respective split value
    return max_sum, best_split_val

#eval given dimension, first gets list of just dimension values to simplify computation
def eval_dimension(dimension_idx, tr_data, labels, num_splits=10):
    dim_val_lst = []
    
    for pt_idx in range(len(tr_data)):
        pt = tr_data[pt_idx]
        dim_val_lst.append(pt[dimension_idx])
    
    max_split_score, best_split_val = get_sum_of_differences(dim_val_lst, labels)    
    
    return best_split_val, max_split_score

#vectorization - simplifies best dimension querying in build_nodes func
#by building a table with each fold's dimensions' scores and split vals
def get_fold_dim_scores(tr_data, labels, num_folds=num_folds_to_test):
    fold_dim_scores = []
    
    for fold_idx in range(num_folds):
        dim_split_scores = []
        dim_split_vals = []  
        
        tr_sub_lst, tst_sub_lst, label_sub_lst = get_sub_lsts(tr_data, labels, fold_idx)
        for dim_idx in range(num_dimensions):
            best_split, max_split_score = eval_dimension(dim_idx, tr_sub_lst, label_sub_lst)
            dim_split_scores.append(max_split_score)
            dim_split_vals.append(best_split)
            
        dim_info = [dim_split_scores, dim_split_vals]
        fold_dim_scores.append(dim_info)
    
    return fold_dim_scores

#get best dimension from whole fold table given above, but for just one fold
def get_best_dimension(dimension_lst, dim_fold_score_lst, fold_num):   
    best_dim_split = 0
    best_dim_split_score = 0
    best_dimension = 0
        
    #determine best dimension given what's remaining in a given dimension lst
    for dimension_idx in dimension_lst:    
        split_score_lst = dim_fold_score_lst[0]
        split_val_lst = dim_fold_score_lst[1] 
        
        max_split_score = split_score_lst[dimension_idx]        
        best_split = split_val_lst[dimension_idx]
        
        if(max_split_score > best_dim_split_score):
            best_dim_split_score = max_split_score
            best_dim_split = best_split
            best_dimension = dimension_idx
    
    return best_dimension, best_dim_split        
    
#get majority vote for given label set, returns whether is unanimous or not
def get_majority_vote(stripped_labels):
    normal_cnt = 0
    fraud_cnt = 0
    
    for label in stripped_labels:
        if(label == "Normal"):
            normal_cnt += 1
        else:
            fraud_cnt += 1
    
    if(normal_cnt > fraud_cnt):
        label =  "Normal"
        is_unanimous = (fraud_cnt == 0)
    else:
        label = "Fraud"
        is_unanimous = (normal_cnt == 0)
    
    return label, is_unanimous, [normal_cnt, fraud_cnt]

#eval decision tree given stripped sub lists, helper function to build tree and return predicted labels
def eval_decision_tree(tr_lst, tst_lst, tr_label_lst, dim_fold_score_lst, max_depth, fold_num, min_leaf_samples, min_winning_margin):
    y_lst = []

    tree = build_tree(tr_lst, tr_label_lst, dim_fold_score_lst, max_depth, fold_num, min_leaf_samples, min_winning_margin)
        
    for test_pt in tst_lst:
        label = predict_label(tree, test_pt)
            
        y_lst.append(label)
    
    return y_lst
    
#this is the core of our build_tree function - recursively builds tree by taking optimal decision at each node
def build_nodes(tr_data, labels, best_dim_lst, max_depth, cur_depth, fold_num, min_leaf_samples, min_winning_margin):
    #get best dimension
    #build node according to best dimension

    make_l_leaf = 0
    make_r_leaf = 0
    
    dim_info = best_dim_lst[cur_depth] #query for best dimension for our current depth
    
    node = dec_tree_node(cur_depth, dim_info) #initialize our node

    ldat, rdat, l_labels, r_labels = split_data(tr_data, labels, dim_info) #split data, get left and right datasets
            
    #print(min_leaf_samples, min_winning_margin)
    #get majority vote for each dataset split
    l_label, l_is_unanimous, l_vote_cnt = get_majority_vote(l_labels)
    r_label, r_is_unanimous, r_vote_cnt = get_majority_vote(r_labels)
        
#determine when to add both right and left leaves
    if(cur_depth == max_depth): #reach max depth, make leaves/label nodes
        make_l_leaf = 1
        make_r_leaf = 1
        
    elif(l_is_unanimous and r_is_unanimous): #if both sides are unanimous, no point in splitting further, add labels
        make_l_leaf = 1
        make_r_leaf = 1
        
    elif(not(min_leaf_samples == None)): #attempt to prevent overfitting
        if(len(ldat)/2 < min_leaf_samples):
            make_l_leaf = 1
            make_r_leaf = 1       
                
#determine when to add either right or left leaves            
    if(not((min_winning_margin) == None)): #minimizing number of unanimous leaf nodes/overfitting
        l_normal_cnt = l_vote_cnt[0]
        l_fraud_cnt = l_vote_cnt[1]
        r_normal_cnt = r_vote_cnt[0]
        r_fraud_cnt = r_vote_cnt[1]
        
        #calculating difference between winner and loser vote counts
        l_diff = (max([l_normal_cnt, l_fraud_cnt]) - min([l_normal_cnt, l_fraud_cnt]))
        r_diff = (max([r_normal_cnt, r_fraud_cnt]) - min([r_normal_cnt, r_fraud_cnt]))
        
        #print(l_diff, r_diff, max([l_normal_cnt, l_fraud_cnt]), max([r_normal_cnt, r_fraud_cnt]))
    #when to add left leaf, right node        
        if(l_diff < min_winning_margin):
            make_l_leaf = 1
    
    #when to add right leaf, left node
        elif(r_diff < min_winning_margin):
            make_r_leaf = 1
    
    #if one is unanimous, make leaf
    if(l_is_unanimous):
        make_l_leaf = 1
    elif(r_is_unanimous):
        make_r_leaf = 1
        
#Make appropriate leaves
    #base case 1: hyperparam limit reached
    if(make_r_leaf and make_l_leaf):
        node.add_r_leaf(r_label)
        node.add_l_leaf(l_label)
        return node
    
    #base case 2: l is unanimous
    if(make_l_leaf):
        node.add_l_leaf(l_label)
        r_node = build_nodes(rdat, r_labels, best_dim_lst, max_depth, cur_depth+1, fold_num, min_leaf_samples, min_winning_margin)  
        
        if(not(r_node == None)):
            node.add_r_child(r_node)
            
        return node
    #base case 3: r is unanimous    
    elif(make_r_leaf):    
        node.add_r_leaf(r_label)
        l_node = build_nodes(ldat, l_labels, best_dim_lst, max_depth, cur_depth+1, fold_num, min_leaf_samples, min_winning_margin)     
        
        if(not(l_node == None)):
            node.add_l_child(l_node)
            
        return node
    #otherwise recurse on both sides
    else:    
        l_node = build_nodes(ldat, l_labels, best_dim_lst, max_depth, cur_depth+1, fold_num, min_leaf_samples, min_winning_margin)
        r_node = build_nodes(rdat, r_labels, best_dim_lst, max_depth, cur_depth+1, fold_num, min_leaf_samples, min_winning_margin)
            
        if(not(l_node == None)):
            node.add_l_child(l_node)
        
        if(not(r_node == None)):
            node.add_r_child(r_node)
            
        return node

#sorts best dimension idx scores for given fold and max depth for simple querying
def build_best_dim_lst(fold_dim_scores, max_depth, fold_num):
    dimension_lst = [*range(0,(num_dimensions),1)]
    best_dim_lst = []
    
    for i in range(max_depth+1):
        dim_info = get_best_dimension(dimension_lst, fold_dim_scores, fold_num)
        best_dim_lst.append(dim_info)
        dimension_lst.remove(dim_info[0])
        
    return best_dim_lst    

#build tree function
def build_tree(tr_data, labels, fold_dim_scores, max_depth, fold_num, min_leaf_samples, min_winning_margin):
    #best dim lst corresponds to lst of ordered best dimensions for a given fold and max depth
    best_dim_lst = build_best_dim_lst(fold_dim_scores, max_depth, fold_num)
    
    #build nodes
    root = build_nodes(tr_data, labels, best_dim_lst, max_depth, 0, fold_num, min_leaf_samples, min_winning_margin)
        
    return root

#predict label by recursively navigating the decision tree we've build
def predict_label(node, test_pt):
    if(test_pt[node.dim_idx] < node.dim_split_val):
        if(node.has_l_leaf):
            return node.l_label_vote
        else:
            return predict_label(node.left_child, test_pt)
    else:
        if(node.has_r_leaf):
            return node.r_label_vote
        else:
            return predict_label(node.right_child, test_pt)

#####################################################
#####################################################
#Common use functions

#function to read training data file into appropriate formatted list
def get_dat_lst(rd_file):
    file_dat = open(rd_file)
    dat_lst = []
    
    for line in file_dat:
        n_features = line.split(",") #splits the columns using comma as delimiter
        el = n_features.pop(len(n_features)-1) #removing newline char from last element in lst
        el = el.rstrip()
        n_features.append(el.rstrip())
            
        for i in range(len(n_features)):
            n_features[i] = float(n_features[i])
        dat_lst.append(n_features)
     
    return dat_lst    

#read in labels and format into readable label lst
def get_label_lst(rd_file):
    file_dat = open(rd_file)
    label_lst = []
    
    for line in file_dat:
        cols = line.split(",") #splits the columns using comma as delimiter
        el = cols.pop(0) #removing newline char from last element in lst
        if(int(float(el.rstrip())) == 0):
            label_lst.append("Normal")
        else:
            label_lst.append("Fraud")
                            
    return label_lst 

#function to get sub lists given larger dataset, used for cross fold validation
def get_sub_lsts(dat_lst, label_lst, fold_num, num_folds=num_folds_to_test):
    train_sub_lst = []
    test_sub_lst = []
    label_sub_lst = []  
    
    #divide length of dat lst by total number of folds to get correct sub list
    sub_lst_len = int(len(dat_lst)/num_folds)
    start_idx = int(sub_lst_len*fold_num)
    stop_idx = int(start_idx+sub_lst_len)
    
    #extracting testing data given our start idx and fold num
    test_sub_lst = dat_lst[start_idx:stop_idx]
    
    #removing unneeded elements from training data
    for i in range(len(dat_lst)):
        if i not in range(start_idx, stop_idx):
            train_sub_lst.append(dat_lst[i])
            label_sub_lst.append(label_lst[i])
            
    return train_sub_lst, test_sub_lst, label_sub_lst    

#calculate f1, precision, recall scores by getting tp, fp, tn, fn counts and computing
def calc_f1_score(pred_labels, acc_labels):
    f1_score_lst = []
    
    for model in range (len(pred_labels)):
        print("Evaluating performance of fold:", model)       
        
        pred_sub_lst_len = len(pred_labels[model])
        true_pos_cnt = 1 #avoid division by 0
        true_neg_cnt = 0
        false_pos_cnt = 1
        false_neg_cnt = 0
        
        for i in range(pred_sub_lst_len):
            acc_label_idx = (model*pred_sub_lst_len)+i
            
            pred_label = pred_labels[model][i]
            acc_label = acc_labels[acc_label_idx]
                        
            if(pred_label == acc_label):
                if(pred_label == "Fraud"):
                    true_pos_cnt +=1
                else:
                    true_neg_cnt +=1
                    
            elif((pred_label == "Fraud") and (acc_label == "Normal")):
                false_pos_cnt += 1
            else:
                false_neg_cnt += 1
                
        model_precision = (true_pos_cnt/(true_pos_cnt+false_pos_cnt))
        model_recall = (true_pos_cnt/(true_pos_cnt+false_neg_cnt))
        model_fpr = (false_pos_cnt/(true_pos_cnt+false_pos_cnt))
        
        f1_score = 2*((model_precision*model_recall)/(model_precision+model_recall))
        
        f1_score_lst.append(f1_score)
        print("F1 Score for fold", model,":", f1_score, ", precision:", model_precision, ", recall: ", model_recall, ", fpr:", model_fpr)
        
    best_model = f1_score_lst.index(max(f1_score_lst))
    max_f1_score = max(f1_score_lst)
    
    print("Best fold for this testcase:", best_model)
    return best_model, max_f1_score

#eval specific fold
def eval_fold(tr_lst, label_lst, dim_fold_score_lst, testcase_val, is_knn, fold_num, min_leaf_samples, min_winning_margin, tst_lst=[]):
    tr_sub_lst, tst_sub_lst, label_sub_lst = get_sub_lsts(tr_lst, label_lst, fold_num)
        
    sub_y_lst = []
    
    if(tst_lst == []):
        lst_to_tst = tst_sub_lst
    else:
        lst_to_tst = tst_lst
    
    if(is_knn):
        num_neighbors = testcase_val
        sub_y_lst = eval_knn_tree(num_neighbors, tr_sub_lst, lst_to_tst, label_sub_lst)
    else:
        max_depth = testcase_val
        sub_y_lst = eval_decision_tree(tr_sub_lst, lst_to_tst, label_sub_lst, dim_fold_score_lst, max_depth, fold_num, min_leaf_samples, min_winning_margin)
            
    return sub_y_lst

#evaluate all folds specified, return labels
def eval_folds(tr_lst, label_lst, dim_score_lst, testcase_val, is_knn, min_leaf_samples=None, min_winning_margin=None, tot_folds=num_folds_to_test):
    pred_y_lst = []
    
    for i in range(tot_folds):        
        begin_time = datetime.datetime.now()       
        sub_y_lst = eval_fold(tr_lst, label_lst, dim_score_lst[i], testcase_val, is_knn, i, min_leaf_samples, min_winning_margin)
        pred_y_lst.append(sub_y_lst)  
        print("Time fold", i, "took: ", (datetime.datetime.now() - begin_time))
        
    return pred_y_lst

#func: iter_testcases
#a function to iterate through testcases for each training type
def iter_testcases(tr_lst, label_lst, testcase_lst, is_knn):
    if(is_knn):
        str_test = "num neighbors"
    else:
        str_test = "max depth"
    
    for testcase in testcase_lst:
        print("Evaluating testcase with", str_test,":", testcase)
        begin_time = datetime.datetime.now()

        if(is_knn):
            fold_dim_scores = [[],[],[],[],[]]
        else:
            fold_dim_scores = get_fold_dim_scores(tr_lst, label_lst)
            
        pred_y_lst = eval_folds(tr_lst, label_lst, fold_dim_scores, testcase, is_knn)   
            
        calc_f1_score(pred_y_lst, label_lst)
        
        print("Time this testcase took: ", (datetime.datetime.now() - begin_time), "\n")
    
    print("Done evaluating testcases...\n")

#runrandom decision tree with params
def run_random_decisionTree(tr_lst, label_lst, param_vals, is_knn=0, tot_folds=num_folds_to_test):
    max_depth = param_vals[0]
    min_leaf_samples = param_vals[1]
    min_winning_margin = param_vals[2]
    pred_y_lst = []

    fold_dim_scores = get_fold_dim_scores(tr_lst, label_lst)
    pred_y_lst = eval_folds(tr_lst, label_lst, fold_dim_scores, max_depth, is_knn, min_leaf_samples, min_winning_margin)   
    best_model, best_f1_score = calc_f1_score(pred_y_lst, label_lst) 
    return best_model, best_f1_score

#apply random parameter search for decision tree model
def random_search_decisionTree(tr_lst, acc_y_lst):
    best_param_vals = []
    best_model = 0
    max_f1_score = 0
    
    #20 random hyperparam combos for decision tree
    for i in range(20):
        max_depth = random.randint(3, num_dimensions)#highest f1 scores acc to prev output
        min_leaf_samples = random.randint(0, 100)
        min_winning_margin = random.randint(0, 20)
        param_vals = [max_depth, min_leaf_samples, min_winning_margin]
        print("Testing parameters:", param_vals)
        mod, f1 = run_random_decisionTree(tr_lst, acc_y_lst, param_vals, 0, tot_folds=num_folds_to_test) 
        
        if(f1 > max_f1_score):
            max_f1_score = f1
            best_model = mod
            best_param_vals = param_vals
    
    return best_model, best_param_vals

#convert string label lst to int vals
def convert_labels(pred_labels, acc_labels):
    pred_y_vals = []
    acc_y_vals = []
    
    for i in range(len(acc_labels)):
        if(pred_labels[i] == "Normal"):
            pred_y_vals.append(0)
        else:
            pred_y_vals.append(1)
            
        if(acc_labels[i] == "Normal"):
            acc_y_vals.append(0)
        else:
            acc_y_vals.append(1)   
    pred_y_vals = np.array(pred_y_vals)
    acc_y_vals = np.array(acc_y_vals)
    
    return pred_y_vals, acc_y_vals
        
#####################################################
#####################################################
def write_pred_to_file(pred_labels, write_file):
        
    with open(write_file, mode='w') as wf:
        for label in pred_labels:
            wf.write(label+"\n")
        
def main():

    data_dir = Path("../../Data") #Specify path to Data directory
    os.chdir(data_dir) #Change working directory to Data directory
 
    tr_lst = get_dat_lst("x_train.csv")
    acc_y_lst = get_label_lst("y_train.csv")
    
#####TRAINING####
    #KNN
    print("Training data using KNN...")
    num_neighbors_to_tst = [3,5,10,20,25]
    iter_testcases(tr_lst, acc_y_lst, num_neighbors_to_tst, 1)

    #DecisionTree
    print("Training data using Decision Tree...")
    num_depths_to_tst = [3,6,9,12,15]
    iter_testcases(tr_lst, acc_y_lst, num_depths_to_tst, 0) 

##Random Search
    #print(random_search_decisionTree(tr_lst, acc_y_lst))

####TESTING####
    #Model choice: DecisionTree; max_depth = 6
    is_knn = 0
    testcase_val = 6
    fold_num = 4    

    fold_dim_scores = get_fold_dim_scores(tr_lst, acc_y_lst)
    
    #running best model on subset of training data
    pred_y_lst = eval_fold(tr_lst, acc_y_lst, fold_dim_scores[fold_num], testcase_val, is_knn, fold_num, None, None)
    calc_f1_score([pred_y_lst], acc_y_lst)
    start_idx = fold_num*len(pred_y_lst)
    sub_acc_y_lst = acc_y_lst[start_idx:(start_idx+len(pred_y_lst))]
    pred, acc = convert_labels(pred_y_lst, sub_acc_y_lst)
    
    print("AUC for best model fold:", metrics.roc_auc_score(acc, pred), "\n")
    
    print("Using selected training model on all our training data...")
    tst_lst = tr_lst 
    
    #running best model on all training data
    pred_y_lst = eval_fold(tr_lst, acc_y_lst, fold_dim_scores[fold_num], testcase_val, is_knn, fold_num, None, None, tst_lst)
    calc_f1_score([pred_y_lst], acc_y_lst)
    pred, acc = convert_labels(pred_y_lst, acc_y_lst)

    print("AUC for all test data:", metrics.roc_auc_score(acc, pred))

    print("Using selected training model on our test data...")
    tst_lst = get_dat_lst("x_test.csv")
    
    data_dir = Path("../Submission/Predictions") #Specify path to Submission directory
    os.chdir(data_dir) #Change working directory to Submission directory
    
    pred_y_lst = eval_fold(tr_lst, acc_y_lst, fold_dim_scores[fold_num], testcase_val, is_knn, fold_num, None, None, tst_lst)
    
    print("Writing test data to csv file...")    
    write_pred_to_file(pred_y_lst, "best.csv")    
    
main()