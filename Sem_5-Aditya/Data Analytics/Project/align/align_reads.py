from align.support import *
from align.rank import SuccinctRank
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


next_char = {'A': 'C', 'C': 'G', 'G': 'T', 'T': '$'}

def align_reads(bwt, n, idx, reference, reads, err_thresh):

    print("Initialising Succinct Rank...")
    rk_A = SuccinctRank(bwt,'A')
    rk_C = SuccinctRank(bwt,'C')
    rk_G = SuccinctRank(bwt,'G')
    rk_T = SuccinctRank(bwt,'T')
    print()

    agg_val = {
        'A': 0,
        'C': rk_A.rank(n), 
        'G': rk_A.rank(n)+rk_C.rank(n), 
        'T': rk_A.rank(n)+rk_C.rank(n)+rk_G.rank(n), 
        '$': rk_A.rank(n)+rk_C.rank(n)+rk_G.rank(n)+rk_T.rank(n)
        }
    
    rk_dict = {'A': rk_A, 'C': rk_C, 'G': rk_G, 'T': rk_T}

    reads = list(enumerate(reads))
    progress = tqdm(total=len(reads))

    for num, read in reads:
        process_read((read, num, idx, reference, rk_dict, agg_val, err_thresh))
        progress.update(1)
    progress.close()