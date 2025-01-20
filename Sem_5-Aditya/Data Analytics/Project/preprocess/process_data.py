import numpy as np

def get_data(bwt_path, map_path, fa_path, reads_path):

    with open(bwt_path, 'r') as file:
        bwt = file.read().replace('\n','')
    n = len(bwt)

    with open(map_path, 'r') as file:
        idx = file.readlines()
        idx = [int(x.strip()) for x in idx]
    
    with open(fa_path, 'r') as file:
        file.readline()
        reference = file.read().replace('\n','')

    reference = np.array(list(reference))

    with open(reads_path, 'r') as file:
        reads = file.readlines()
    
    return bwt, n, idx, reference, reads 



    