# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:32:48 2018

@author: jdkan
"""
import time
import alignment
import copy
import functools
# Your task is to *accurately* predict the primer melting points using machine 
# learning based on the sequence of the primer.

# Load the primers and their melting points.


@functools.cache
def melting_point(primer, melting_point_rf):
    return melting_point_rf.predict([CalculatePrimerFeatures(primer)])

def CalculatePrimerFeatures(seq):
    # modify this function to return a python list of feature values for a given sequence for Task 1
    return [len(seq), seq.count("A"), seq.count("T"), seq.count("G"), seq.count("C")]

def PredictPCRProduct(primer1, primer2, template_sequence, melting_point_rf):
    """
    Input:
        primer1 = a primer sequence in 5' to 3' order
        primer2 = a primer sequence in 5' to 3' order
        template_sequence = sequence from which we are trying to generate 
        copies using PCR in 5' to 3' order.  Assume this is double stranded, 
        but we are only including the top strand in the argument.
        melting_point_rf = random forest learned from task1 to predict primer
        melting points.
    Output:
        return sequence of successful PCR amplication reaction or None (if 
        there is no successful reaction)
        
    """
    
    if (len(primer1) < 18 or len(primer1) > 35):
        return None
    if (len(primer2) < 18 or len(primer2) > 35):
        return None
    melting_point1 = melting_point_rf.predict([CalculatePrimerFeatures(primer1)])
    melting_point2 = melting_point_rf.predict([CalculatePrimerFeatures(primer2)])
    
    if (abs(melting_point1 - 60) > 2 or abs(melting_point2 - 60) > 2):
        return None
    
    pairs = {
        "T": "A",
        "A": "T",
        "C": "G",
        "G": "C",
        "N": "N"
    }
    
    compliment = "".join(list(map(lambda x : pairs[x], template_sequence)))
    
    #p2r == c && p1 == t
    #p1r == c && p2 == t
    
    p1t_score = alignment.local_align(primer1, template_sequence)[0] / (10 * len(primer1))
    p1rc_score = alignment.local_align(primer1[::-1], compliment)[0] / (10 * len(primer1))
    p2t_score = alignment.local_align(primer2, template_sequence)[0] / (10 * len(primer2))
    p2rc_score = alignment.local_align(primer2[::-1], compliment)[0] / (10 * len(primer2))
    
    if (p1t_score >= .8 and p2rc_score >= .8):
        pos_left = alignment.local_align(primer1, template_sequence)[1][1]
        pos_right = alignment.local_align(primer2[::-1], compliment)[1][1]
        print("pos_left: ", pos_left, " pos_right", pos_right)
        if (pos_left < pos_right):
            return template_sequence[pos_left - len(primer1):pos_right]
        
    if (p2t_score >= .8 and p1rc_score >= .8):
        pos_left = alignment.local_align(primer2, template_sequence)[1][1]
        pos_right = alignment.local_align(primer1[::-1], compliment)[1][1]
        print("pos_left: ", pos_left, " pos_right", pos_right)
        if (pos_left < pos_right):
            return template_sequence[pos_left - len(primer2):pos_right]
    
   
    return None




def LoadFastA(path):
    infile = open(path, 'r')
    seq = ""
    infile.readline()
    for line in infile:
        seq += line[:-1]
    return seq

task2_randomforest = None
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   from sklearn.metrics import r2_score
   from sklearn.ensemble import RandomForestRegressor
   import importlib
   import os
   import sys

   print("Running Task 1:")
   
   infile = open("training_primers.txt", 'r')
   infile.readline() # don't load headers
   primers = []
   melting_points = []
   features = []
   st = time.time()
   for line in infile:
       Line = line.split()
       primers.append(Line[0])
       melting_points.append(float(Line[1]))
       # calculate features
       features.append(CalculatePrimerFeatures(Line[0]))
   feat_time = (time.time()-st)/(len(features)/1000)
   # cross validation
   how_many_folds = 10 
   predictions = []
   truth = []
   my_len = len(features[-1])
    
   for fold in range(how_many_folds):
        #print ("Calculating Fold",fold)
        training_features = []
        training_outcomes = []
        testing_features = []
        testing_outcomes = []
        for c in range(len(melting_points)):
            if c % how_many_folds == fold:
                # put this one in testing data
                testing_features.append(features[c])
                testing_outcomes.append(melting_points[c])
            else:
                # put this one in training data
                training_features.append(features[c])
                training_outcomes.append(melting_points[c])
        # train the model
        
        rf = RandomForestRegressor(n_estimators = 200)
        rf.fit(training_features, training_outcomes)
        fold_predictions = rf.predict(testing_features)
        truth += testing_outcomes
        predictions += list(fold_predictions)
      
           
   #truth = np.array(truth)
   #predictions = np.array(predictions)        
   print("Task 1 Results:\n")
   print("R2 Score:", r2_score(truth, predictions))
   
   """
   Task 2:
   Design a function to predict whether a product will be made in a PCR reaction.
   Your function should take as input the template DNA and the two primers and 
   return the product or 'None'.

   This requires a local alignment function which is provided for you or you 
   can use another implementation.
    
   There are test cases in PCR_product_test_cases.txt.
   
   correct temperature
   >80% alignment
   primer reverse matches with original DNA
   
   """
   task2_randomforest = RandomForestRegressor(n_estimators = 200)
   task2_randomforest.fit(features, melting_points)
   
   PredictPCRProduct("AACTACGGAGAACTACAGCAACCT", "TGGTGGGATGTCTTTCAACAGG",  "ACGTCAGCGAGCGCTACGACGTGGTGGGATGTCTTTCAACAGGACGGACTGACGCGACGACTGACTGTAGGCTAGGTTGCTGTAGTTCTCCGTAGTTAGCTACGACGCATGCAGCTGCA", task2_randomforest)
   
   
   """
   Task 3:
   Design primers for a PCR reaction to distinguish between the three types 
   of DNA.  
   
   -Your primers should be between 18 and 35 bases long.  
   -They should have at least 80% match to the DNA strand.
   -Predicted melting points of any primers to be run in the same reaction
   should be between 58.0 and 62.0 C.
   -Products are distinguishable in length if their difference in length is >40
   bases
   -Your products should not be longer than 1000 bases.
   
   We are making predictions about the functionality of sets of primers.  We 
   will synthesize your group's primers and test them in the lab later.
     
   
   """
   

def get_primers_to_diff(DNA : list):
    short = min(DNA)
    pairs = {
        "T": "A",
        "A": "T",
        "C": "G",
        "G": "C"
    }
    c = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0 
    c6 = 0
    n2 = 0
    # i = [0,17], [285,386]
    for i in range(283, len(short) - 80):
        print("I : ", i)
        for j in range(i + 19,i + 37):
            p1 = short[i:j]
            if ("N" in p1):
                continue
            if (abs(melting_point(p1, task2_randomforest)-60) > 2):
                continue
            print("J : ", j)
            for k in range(j + 20, len(short) - 40):
                for l in range(k + 19, k + 37):
                    c += 1
                    if(c % 1000 == 0):
                        print("Count: ", c, " Progress: ", (i * 18 + (j - i - 18)) * 100 /(len(short) - 80), "%", "c1: ", c1, "c2: ", c2, "c3: ", c3, "c4: ", c4, "c5: ", c5, "c6: ", c6, "n2: ", n2)
                    p2 = short[k:l]
                    if ("N" in p2):
                        continue
                   
                    p2r = p2[::-1]
                    p2rc = "".join(list(map(lambda x : pairs[x], p2r)))
                    if (abs(melting_point(p2rc, task2_randomforest)-60) > 2):
                        continue
                    prod1 = PredictPCRProduct(p1, p2rc, DNA[0], task2_randomforest)
                    
                    if (prod1):
                        c1 += 1
                        prods = [prod1] + list(map(lambda x : PredictPCRProduct(p1, p2rc, x, task2_randomforest), DNA[1:]))
                        count = sum(list(map(lambda x : 1 if x else 0, prods)))
                        if (prods[1]):
                            c2 += 1
                        if (prods[2]):
                            c3 += 1
                        if (prods[3]):
                            c4 += 1
                        if (prods[4]):
                            c5 += 1
                        if (prods[5]):
                            c6 += 1
                            
                        if (count == 2):
                            n2 += 1
                        if (count >= 3):
                            filtered = filter(None,prods)
                            lens = list(map(len, filtered))
                            
                            if(max(lens) - min(lens) >= 40 and min(lens) > 40 and max(lens) < 1000): 
                                print(lens)
                                # for le in lens:
                                #     if (abs(le - max(lens)) >= 40 and abs(le - min(lens) >= 40)):
                                print("p1: ", p1)
                                print(melting_point(p1, task2_randomforest))
                                print("p2rc: ", p2rc)
                                print(melting_point(p2rc, task2_randomforest))
                                print(list(map(lambda x : 1 if x else 0, prods)))
                                return (p1, p2rc)
                    
DNA = [
    "catgctcagattgacgctgcggcaggcttaacacatgcaagtcgagcggggatagggtgcttgcnnngattcctagcggcggacgggtgagtaatgcttaggaatctgcctattagtgggggacaacgttccgaaagggacgctaataccgcatacgtcctacgggagaaagcaggggatcttcggaccttgcgctaatagatgagcctaagtcggattagctagttggtggggtaaaggcctaccaaggcgacgatctgtagcgggtctgagaggatgatccgccacactgggactgagacacggcccagactcctacgggaggcagcagtggggaatattggacaatggggggaaccctgatccagccatgccgcgtgtgtgaagaaggccttttggttgtaaagcactttaagcgaggaggaggcttacctggttaatacctgggataagtggacgttactcgcagaataagcaccggctaactctgtgccagcagccgcggtaatacagagggtgcaagcgttaatcggatttactgggcgtAaagcgcgcgtaggtggctaattaagtcaaatgtgaaatccccgagcttaacttgggaattgcattcgatactggttagctagagtatgggagaggatggtagaattccaggtgtagcggtgaaatgcgtagagatctggaggaataccgatggcgaaggcagccatctggcctaatactgacactgaggtgcgaaagcatggggagcaaacaggattagataccctggtagtccatgccgtaaacgatgtctactagccgttggggcccttgaggctttagtggcgcagctaacgcgataagtagaccgcctggggagtacggtcgcaagactaaaactcaaatgaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgatgcaacgcgaagaaccttacctggccttgacatacagagaactttccagagatggattggtgccttcgggaactctgatacaggtgctgcatggctgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttttccttatttgccagcacttcgggtgggaactttaaggatactgccagtgacaaactggaggaaggcggggacgacgtcaagtcatcatggcccttacggccagggctacacacgtgctacaatggtcggtacaaagggttgctactgcgcgagcagatgctaatctcaaaaagccgatcgtagtccggatcgcagtctgcaactcgactgcgtgaagtcggaatcgctagtaatcgcggatcagaatgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccatgggagtttgttgcaccagaagtaggtagtctaacctt",
    "tacatgcaagtcgagcgaactgacgaggagcttgctcctttgacgttagcggcggacgggtgagtaacacgtgggtaacctacctataagactggaataactccgggaaaccggggctaatgccggataacatgttgaaccgcatggttcaacattgaaaggcggttttgctgtcacttatagatggacctgcgccgtattagctagttggtnaggtaatggcttaccaaggcgacgatacgtagccgacctgagagggtgatcggccacactggaactgagacacggtccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgatgaaggttttcggatcgtaaagctctgttattagggaagaacaagtgcgtaggtaactatgcgcaccttgacggtacctaatcagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttatccggaattattgggcgtaaagcgcgcgtaggcggtttcttaagtctgatgtgaaagcccacggctcaaccgtggatggtcattggaaactggggaacttgagtgcagaagaggaaagtggaattccatgtgtagcggtgaaatgcgcagagatatggaggaacaccagtggcgaaggcgactttctggtctgtaactgacgctgatgtgcgaaagcgtggggatcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagggggtttccgccccttagtgctgcagctaacgcattaagcactccgcctggggagtacgatcgcaagattgaaactcaaaggaattgacggggacccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaaatcttgacatcctttgatcgctctagagatagagttttccccttcgggggacaaagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttaagcttagttgccatcattaagttgggcactctaagttgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgatttgggctacacacgtgctacaatggacggtacaaagggtcgctaaaccgcgaggtcaagcaaatcccataaagccgttctcagttcggattgtagtctgcaactcgactacatgaagctggaatcgctagtaatcgtagatcagcatgctacggtgaatacnttcccgggtcttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagccggtggagtaacctttggagctagccgtcga",
    "tcagattgacgctggcggcaggcctaacacatgcaagtcgagcggatgagaggagcttgctcctcgattcagcggcggacgggtgagtaatgcctaggaatctgcctagtagtgggggacaacgtttcgaaaggaacgctaataccgcatacgtcctacgggagaaagtgggggatcttcggacctcacgctattagatgagcctaggtcggattagctagttggtagggtaaaggcctaccaaggcgacgatccgtaactggtctgagaggatgatcagtcacactggaactgagacacggtccagactcctacgggaggcagcagtggggaatattggacaatgggcgaaagcctgatccagccatgccgcgtgtgtgaagaaggccttcgggtcgtaaagcactttaagttgggaggaagggctcatagcgaatacctgtgagttttgacgttaccaacagaataagcaccggctaacttcgtgccagcagccgcggtaatacgaagggtgcaagcgttaatcGgAattactgggcgtaaagcgcgcgtaggtggcttgataagttggatgtgaaatccccgggctcaacctgggaactgcatccaaaactgtctggctagagtgcggtagagggtagtggaatttccagtgtagcggtgaaatgcgtagatattggaaggaacaccagtggcgaaggcgactacctggactgacactgacactgaggtgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgtcaactagccgttgggatccttgagatcttagtggcgcagctaacgcattaagttgaccgcctggggagtacggccgcaaggttaaaactcaaatgaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttacctggccttgacatgctgagaactttccagagatggattggtgccttcgggaactcagacacaggtgctgcatggctgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgtaacgagcgcaacccttgtccttagttaccagcacgttatggtgggcactctaaggagactgccggtgacaaaccggaggaaggtggggatgacgtcaagtcatcatggcccttacggccagggctacacacgtgctacaatggtcggtacaaagggttgccaagccgcgaggtggagctaatcccataaaaccgatcgtagtccggatcgcagtctgcaactcgactgcgtgaagtcggaatcgctagtaatcgtgaatcanaacgtcacggtgaatacgttcccngggccttgtacacaccgcccgtcacaccatgggagtgggttgctccagaagtagctagtctaaccctcgg",
    "atgacgctgcgcgtgcctaatacatgcaagtcgagcgaatggattaagagcttgctcttatgaagttagcggcggacgggtgagtaacacgtgggtaacctgcccataagactgggataactccgggaaaccggggctaataccggataatattttgaactgcatggttcgaaattgaaaggcggcttcggctgtcacttatggatggacccgcgtcgcattagctagttggtgaggtaacggctcaccaaggcaacgatgcgtagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgatgaaggctttcgggtcgtaaaactctgttgttagggaagaacaagtgctagttgaataagctggcaccttgacggtacctaaccagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttatccggaattattgggcgtaaagcgcgcgcaggtggtttcttaagtctgatgtgaaagcccacggctcaaccgtggagggtcattggaaactgggagacttgagtgcagaagaggaaagtggaattccatgtgtagcggtgaaatgcgtagagatatggaggaacaccagtggcgaaggcgactttctggtctgtaactgacactgaggcgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagagggtttccgccctttagtgctgaagttaacgcattaagcactccgcctggggagtacggccgcaaggctgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcctctgaaaaccctagagatagggcttctccttcgggagcagagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttgatcttagttgccatcattaagttgggcactctaaggtgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggacggtacaaagagctgcaagaccgcgaggtggagctaatctcataaaaccgttctcagttcggattgtaggctgcaactcgcctacatgaagctggaatcgctagtaatcgcggatcagcatgccgcggtgaatacgttccccgggccttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagtcngtggggtaacctttatggagccagccgccta",
    "acgctgcgcgtgcctatacatgcaagtcgagcgaatggattgagagcttgctctcaagaagttagcggcggacgggtgagtaacacgtgggtaacctgcccataagactgggataactccgggaaaccggggctaataccggataacattttgaactgcatggttngaaattgaaaggcggcttcggctgtcacttatggatggacccgcgtcgcattagctagttggtgaggtaacggctcaccaaggcaacgatgcgtagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgatgaaggctttcgggtcgtaaaactctgttgttagggaagaacaagtgctagttgaataagctggcaccttgacggtacctaaccagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttatccggaattattgggcgtaaagcgcgcgcaggtggtttcttaagtctgatgtgaaagcccacggctcaaccgtggagggtcattggaaactgggagacttgagtgcagaagaggaaagtggaattccatgtgtagcggtgaaatgcgtagagatatggaggaacaccagtggcgaaggcgactttctggtctgtaactgacactgaggcgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagagggtttccgccctttagtgctgaagttaacgcattaagcactccgcctggggagtacggccgcaaggctgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcctctgaaaaccctagagatagggcttctccttcgggagcagagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttgatcttagttgccatcattaagttgggcactctaaggtgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggacggtacaaagagctgcaagaccgcgaggtggagctaatctcataaaaccgttctcagttcggattgtaggctgcaactcgcctacatgaagctggaatcgctagtaatcgcggatcagcatgccgcggtgaatacnttccngnccttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagtcngtggggtaacctttttggagccagccgccta",
    "catgctcagaacgacgctgcggcatgcctaatacatgcaagtcgaacgatcctttcggggatagtggcgcacgggtgcgtaacgcgtgggaatctgcccntngggttcggaataacttcgggaaactgaagctaataccggatgatgacgaaagtccaaagatttatcgcccagggatgagcccgcgtaggattagctagttggtggggtaaaggcctaccaaggcgacgatccttagctggtctgagaggatgatcagccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatattggacaatgggcgaaagcctgatccagcaatgccgcgtgagtgatgaaggccttagggttgtaaagctcttttacccgagatgataatgacagtatcgggagaataagctccggctaactccgtgccagcagccgcggtaatacggagggagctagcgttgttCGgAattactgggcgtAaagcgcacgtaggcggcgatttaagtcagaggtgaaagcccggggctcaaccccggaactgcctttgagactggattgctagaatcttggagaggcgagtggaattccgagtgtagaggtgaaattcgtagatattcggaagaacaccagtggcgaaggcggctcgctggacaagtattgacgctgaggtgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgataactagctgctggggcacatggtgtttcggtggcgcagctaacgcattaagttatccgcctggggagtacggtcgcaagattaaaactcaaaggaattgacgggggcctgcacaagcggtggagcatgtggtttaattcgaagcaacgcgcagaaccttaccagcgtttgacatcctcatcgcggatttcagagatgatttccttcagttcggctggatgagtgacaggtgctgcatggctgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaaccctcgcctttagttgccagcattcagttgggtactctaaaggaaccgccggtgataagccggaggaaggtggggatgacgtcaagtcctcatggcccttacgcgctgggctacacacgtgctacaatggcgactacagtgggctgcaaccgtgcgagcggtagctaatctccaaaagtcgtctcagttcggattgttctctgcaactcgagagcatgaaggcggaatcgctagtaatcgcggatcagcatgccgcggtgaatacgttcccnngccttgtacacaccgcccgtcacaccatgggatttggattcacccganncactgc"
    ]

get_primers_to_diff(list(map(lambda x : x.upper(), DNA)))

def reverse_comp(x):
    pairs = {
        "T": "A",
        "A": "T",
        "C": "G",
        "G": "C"
    }
    r = x[::-1]
    rc = "".join(list(map(lambda x : pairs[x], r)))
    return rc


# [0,49,0,736,736,49]
p1 = "CCACACTGGGACTGAGACA".upper()
p1_norm = p1.lower()
p2 = "TACTCTGCTCCCGAAGGAG"
p2_norm = reverse_comp(p2).lower()

# [979, 136, 136, 979, 979, 955]
# p1 = "GCCGCGTGTGTGTTGAAG".upper()
# p1_norm = p1.lower()
# p2 = "ATTGACCGCGGCATGCTG"
# p2_norm = reverse_comp(p2).lower()



print(melting_point(p1, task2_randomforest))
print(melting_point(p2, task2_randomforest))

print("1", len(PredictPCRProduct(p1,p2,DNA[0].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[0].upper(), task2_randomforest) else 0)
print("2", len(PredictPCRProduct(p1,p2,DNA[1].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[1].upper(), task2_randomforest) else 0)
print("3", len(PredictPCRProduct(p1,p2,DNA[2].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[2].upper(), task2_randomforest) else 0)
print("5", len(PredictPCRProduct(p1,p2,DNA[3].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[3].upper(), task2_randomforest) else 0)
print("6", len(PredictPCRProduct(p1,p2,DNA[4].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[4].upper(), task2_randomforest) else 0)
print("7", len(PredictPCRProduct(p1,p2,DNA[5].upper(), task2_randomforest)) if PredictPCRProduct(p1,p2,DNA[5].upper(), task2_randomforest) else 0)

