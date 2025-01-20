import math
import numpy as np

# class to store the rank structure
class SuccintRank:
    def __init__(self, bwt, block_size=32):
        self.bwt = bwt
        self.bwt_len = len(bwt)
        self.block_size = block_size
        self.num_blocks = self.bwt_len // block_size + 1
        self.rank_tables()
        self.count()
    
    # create the rank table to store the rank of each base    
    def rank_tables(self):
        self.rank_table = np.zeros((self.num_blocks, 4), dtype=int)
        for i in range(1, self.num_blocks):
            start = (i-1) * self.block_size
            end = i * self.block_size
            for j, char in enumerate('ACGT'):
                self.rank_table[i, j] = self.rank_table[i-1, j]
                self.rank_table[i, j] += self.bwt[start:end].count(char)
    
    # calculate the rank of a base at a given position
    def rank(self, i, char):
        block = i // self.block_size
        offset = i % self.block_size
        
        block_rank = self.rank_table[block, 'ACGT'.index(char)]
        inside_rank = self.bwt[block*self.block_size : block*self.block_size + offset].count(char)
        return block_rank + inside_rank
    
    # calculate the count of each base in the bwt
    def count(self):
        count_A = self.rank(self.bwt_len, 'A') + (self.bwt[-1] == 'A')
        count_C = self.rank(self.bwt_len, 'C') + (self.bwt[-1] == 'C')
        count_G = self.rank(self.bwt_len, 'G') + (self.bwt[-1] == 'G')
        
        counts = [count_A, count_C, count_G]
        
        # use the counts to calculate the starting point of each base in first column 
        self.bias = [0, sum(counts[:1]), sum(counts[:2]), sum(counts[:3])]


# find the valid positions of a read in the reference genome
def find(read):
    len_subread = len(read) // 3

    # divide into 3 subreads
    # if one of them is an exact match, brute force the other two subreads
    # total mismatches allowed is 2
    subreads = [read[:len_subread], read[len_subread:2*len_subread], read[2*len_subread:]]
    subread_match = []
    
    for subread in subreads:
        start, end = 0, rank_struct.bwt_len - 1

        for char in subread[::-1]:
            bias = rank_struct.bias['ACGT'.index(char)]
            
            start = rank_struct.rank(start, char) + bias
            end = rank_struct.rank(end, char) - (char != rank_struct.bwt[end]) + bias
            
            if start > end:
                start = None
                break
            
        if start is not None:
            matches = [idx[i] for i in range(start, end+1)]
        else:
            matches = []
        
        subread_match.append(matches)
    
    valids = set()
    for i in range(3):
        bias = i * len_subread
        for j in subread_match[i]:
            start = j - bias
            mismatch = sum(1 for a,b in zip(ref[start:start+len(read)], read) if a != b)
            
            if mismatch <= 2:
                valids.add(start)
                
    return valids


# check if the read lies in the red or green exon intervals
def match(valids, interval):
    for start in valids:
        for i in range(6):
            red_range = range(interval[i][0], interval[i][1] + 1)
            if start in red_range:
                return i
    return None


# main function to match reads to the reference genome
def exon_match(reads):
    red_counts, green_counts = [0]*6, [0]*6
    
    for read in reads:
        read = np.array(list(read.strip().replace('N','A')))
        valids = find(read)
        
        # if no valid positions found, try the reverse complement
        if len(valids) == 0:
            read = [complement[x] for x in read][::-1]
            valids = find(read)
        
        red_pos = match(valids, red_int)
        green_pos = match(valids, green_int)
        
        # if the read lies in both red and green intervals, count it as 0.5 for each
        # else count it as 1 for the interval it lies in
        if red_pos is not None and green_pos is not None:
            red_counts[red_pos] += 0.5
            green_counts[green_pos] += 0.5
        elif red_pos is not None and green_pos is None:
            red_counts[red_pos] += 1
        elif red_pos is None and green_pos is not None:
            green_counts[green_pos] += 1
        
    return (red_counts, green_counts)


# calculate the log probability of each configuration
def calculate_prob(counts):
    red = counts[0][1:5]
    green = counts[1][1:5]
    
    log_probs = []
    for i in range(4):
        if 0 in config_red[i]:
            log_probs.append(-1*math.inf)
            continue
        
        log_prob = 0
        for j in range(4):
            log_prob += red[j] * math.log(config_red[i][j])
            log_prob += green[j] * math.log(1 - config_red[i][j])
        
        log_probs.append(log_prob)
    
    return log_probs
    

# intervals for red and green exons
red_int = [[149249757,149249868], [149256127,149256423], [149258412,149258580],
           [149260048,149260213], [149261768,149262007], [149264290,149264400]]

green_int = [[149288166,149288277], [149293258,149293554], [149295542,149295710],
             [149297178,149297343], [149298898,149299137], [149301420,149301530]]

# reverse complement dictionary
complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# probability of each configuration for red exons
config_red = [[0.33, 0.33, 0.33, 0.33], [0.5, 0.5, 0, 0],
              [0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.25, 0.5]]


if __name__ == '__main__':
    with open('../data/chrX_last_col.txt', 'r') as f:
        bwt = f.read().replace('\n','')

    with open('../data/reads', 'r') as f:    
        reads = f.readlines()

    with open('../data/chrX_map.txt', 'r') as f:
        idx = f.readlines()
        idx = [int(x.strip()) for x in idx]

    with open('../data/chrX.fa', 'r') as f:
        f.readline()
        ref = f.read().replace('\n','')

    # create the rank structure with block size 32
    rank_struct = SuccintRank(bwt, block_size=32)

    # run the exon_match function to find the counts of reads in red and green exons
    counts = exon_match(reads)
    print("Red exon counts:", counts[0])
    print("Green exon counts:", counts[1], "\n")
    
    # calculate the probability of each configuration
    log_probs = calculate_prob(counts)
    print("Log probabilities of each configuration:", log_probs)
    print("Most likely configuration:", np.argmax(log_probs) + 1)
