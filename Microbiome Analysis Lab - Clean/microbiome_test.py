# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:32:48 2018

@author: jdkan
"""
import time
import alignment
import copy
import random
import pickle
import matplotlib.pyplot as plt
import time

from tqdm import tqdm




def Load16SFastA(path, fraction = 1.0):
    # from a file, read in all sequences and store them in a dictionary
    # sequences can be randomly ignored for testing by adjusting the fraction
    random.seed(11)
    
    infile = open(path, 'r')
    sequences_16s = {}
    c = 0
    my_seq = ""
    for line in infile:
        if ">" in line:
            my_id = line[1:-1]
            if random.random() < fraction:
                sequences_16s[my_id] = ""
            
            
        else:
            if my_id in sequences_16s:
                sequences_16s[my_id] += line[:-1]
    
       
    return sequences_16s



def ConvertLibaryToKmerSets(library, K=2):
    
    new_lib = {}
    c = 0
    for k in library.keys():
        new_lib[k] = set()
        
        s = library[k]
        
        for i in range(len(s) - K + 1):
            new_lib[k].add(s[i:i+K])
        # add your code here to build the k-mer set
        
    return new_lib

def JaccardIndex(s1, s2):
    numerator = float(len(s1.intersection(s2)))
    denominator = float(len(s1.union(s2)))
    return numerator/denominator

def KmerMatch(sequence_kmer_set, library_kmer_set):
    best_score = 0.0
    best_match = None
    
    #add your code here to find the best kmer match
    
    for key, kmer_set in list(library_kmer_set.items()):
        score = JaccardIndex(sequence_kmer_set, kmer_set)
        if (score > best_score):
            best_score = score
            best_match = key
    return best_score, best_match


def AlignmentMatch(sequence, library):
    best_score = -10000000000
    best_match = None
    
    #add your code here to find the best match using alignment
    
    for key, lib_seq in tqdm(list(library.items()), leave=False):
        score, _, _ = alignment.local_align(sequence, lib_seq)
        if score > best_score:
            best_score = score
            best_match = key
    
    return best_score, best_match

def randomize(sequence, p_error):
    new_seq = list(sequence)
    D = {
        "C": ["A", "T", "G"],
        "A": ["C", "T", "G"],
        "T": ["A", "C", "G"],
        "G": ["A", "T", "C"],
        "N": ["C", "A", "T", "G"]
    }
    for i in range(len(sequence)):
        if random.random() < p_error:
            # print(sequence)
            if(sequence[i] in D["N"]):
                new_seq[i] = random.choice(D[sequence[i]])
            else: new_seq[i] = random.choice(D["N"])
    return "".join(new_seq)



if __name__ == "__main__":
    
    random.seed(12) 
    # baseline_dict = pickle.load(open("baseline.pkl", "rb"))
    
    # score, result = AlignmentMatch("cctaatacatgcaagtcgagcgaatggattaagagcttgctcttatgaagttagcggcggacgggtgagtaacacgtgggtaacctgcccataagactgggataactccgggaaaccggggctaataccggataacattttgaaccgcatggttcgaaattgaaaggcggcttcggctgtcacttatggatggacccgcgtcgcattagctagttggtgaggtaacggctcaccaaggcaacgatgcgtagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttccgcaatggacgaaagtctgacggagcaacgccgcgtgagtgatgaaggctttcgggtcgtaaaactctgttgttagggaagaacaagtgctagttgaataagctggcaccttgacggtacctaaccagaaagccacggctaactacgtgccagcagccgcggtaatacgtaggtggcaagcgttatccggaattattgggcgtaaagcgcgcgcaggtggtttcttaagtctgatgtgaaagcccacggctcaaccgtggagggtcattggaaactgggagacttgagtgcagaagaggaaagtggaattccatgtgtagcggtgaaatgcgtagagatatggaggaacaccagtggcgaaggcgactttctggtctgtaactgacactgaggcgcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaagtgttagagggtttccgccctttagtgctgaagttaacgcattaagcactccgcctggggagtacggccgcaaggctgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcctctgaaaaccctagagatagggcttctccttcgggagcagagtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccttgatcttagttgccatcattaagttgggcactctaaggtgactgccggtgacaaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggacggtacaaagagctgcaagaccgcgaggtggagctaatctcataaaaccgttctcagttcggattgtaggctgcaactcgcctacatgaagctggaatcgctagtaatcgcggatcagcatgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagtcggtggggta", baseline_dict)
    
    # print(result)
    # print(score)

    # 1/0

    for E, p_error in [("", 0)]:
        

        
        fn = "bacterial_16s_genes.fa"
        sequences_16s = Load16SFastA(fn, fraction = 1.0)
        print ("Loaded %d 16s sequences." % len(sequences_16s))
        
    
    
        all_sequences = list(sequences_16s.items())
        
        random.shuffle(all_sequences)
        
        
        
        database_list = all_sequences[:200]
        query_list = all_sequences[200:250]
        
        library_dict = dict(database_list)
        query_dict = dict(query_list)
        
        mutated = dict()
        
        
        for key, lib_seq in tqdm(list(query_dict.items())):
            mutated[key] = randomize(lib_seq, p_error)
        
        query_dict = mutated
        
        baseline_dict = dict()
        try:
            baseline_dict = pickle.load(open("baseline" + E + ".pkl", "rb"))
        except FileNotFoundError:
            t0 = time.time()
            for key, quer_seq in tqdm(list(query_dict.items())):
                score, best = AlignmentMatch(quer_seq, library_dict)
                baseline_dict[key] = best
            align_time = time.time() - t0
            pickle.dump(baseline_dict, open("baseline" + E + ".pkl", "wb"))
            
        
        print("baseline computed")
        
        Ks = [1,3,5,7,9,11,13,15,17,19]
        
        kmer_dict = dict();
        
        kmer_times = dict()
        try:
            kmer_dict = pickle.load(open("kmer_dict" + E + ".pkl", "rb"))
        except FileNotFoundError:
            for K in Ks:
                print(K)
                t0 = time.time()
                kmer_dict[K] = dict()
                kmer_database = ConvertLibaryToKmerSets(library_dict, K)
                kmer_queries = ConvertLibaryToKmerSets(query_dict, K)
                for key, quer_set in tqdm(list(kmer_queries.items())):
                    score, best = KmerMatch(quer_set, kmer_database)
                    kmer_dict[K][key] = best
                kmer_times[K] = time.time() - t0
            pickle.dump(kmer_dict, open("kmer_dict" + E + ".pkl", "wb"))
        
        print("kmers computed")
        
    
        agreement = []
        best_K = 0
        best = 0
        
        for K in kmer_dict.keys():
            print(K)
            total = 0
            agreed = 0
            result = kmer_dict[K]
            for key, match in list(result.items()):
                if baseline_dict[key] == match:
                    agreed += 1
                total += 1
            agreement.append(agreed / total)
            if agreed / total > best:
                best_K = K
                best = agreed / total
        
        categories = ["local alignment", "best kmers"]
        values = [align_time, kmer_times[best_K]]
        
        plt.bar(categories, values, color=['#1f77b4', '#ff7f0e'])
        plt.yscale('log')
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.title('Computation time per method')
        plt.show()

        
        
        print("Best agreement for", p_error, ":", best)
        
        plt.plot(Ks, agreement, label=f"Error: {p_error*100}%")

    plt.legend()
    plt.xlabel("K-mer Size (K)")
    plt.ylabel("Agreement with Local Alignment")
    plt.title("K-mer Agreement vs Error Rate")
    plt.show()
    
    
    
    