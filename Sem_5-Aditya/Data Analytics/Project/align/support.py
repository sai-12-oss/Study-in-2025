from align.edit_dist import edit_distance
import numpy as np
import pickle

rev_map = {"A":"T", "T":"A", "C":"G", "G":"C"}

def reverse_complement(read):
    return ''.join(map(lambda x: rev_map[x], read[::-1]))


def search(read, rk_dict, agg_val, l_r):
    starting_point = agg_val[read[-1]]
    ending_point = l_r - 1
    i = len(read) - 1

    while i>=0 and starting_point <= ending_point:
        char = read[i]
        rk = rk_dict[char]
        starting_point = agg_val[char] + rk.rank(starting_point)
        ending_point = agg_val[char] + rk.rank(ending_point  + 1) - 1
        i -= 1
    return (starting_point, ending_point) if starting_point <= ending_point else None 

def process(read, idx, reference, rk_dict, agg_val, err_thresh):
    l_r = len(reference)
    min_errs = np.inf
    best_start_pos = -1
    method = None
    bad_poses = None
    l = len(read)
    splits = [read[:l//3], read[l//3:2*l//3], read[2*l//3:]]

    for index, split in enumerate(splits):
        band = search(split, rk_dict, agg_val, l_r)
        if not band:
            continue

        matches = idx[band[0]:band[1]+1]

        for pos in matches:
            if pos > 151100500:
                continue

            if index == 0:
                str1, str2 = read[l//3:], reference[pos+len(split):pos+l]

            elif index == 1:
                str1 = read[:l//3] + read[2*l//3:]
                str2 = np.concatenate((reference[pos-len(splits[0]):pos], reference[pos+len(splits[1]):pos+len(splits[1])+len(splits[2])]))

            else:
                str1, str2 = read[:2*l//3], reference[pos-len(splits[0])-len(splits[1]):pos]

            bads = [(in_read != in_ref) for in_read, in_ref in zip(str1, str2)]
            mismatches = sum(bads)

            if mismatches <= err_thresh and mismatches < min_errs:
                min_errs = mismatches
                best_start_pos = pos - (len(splits[0]) if index > 0 else 0) - (len(splits[1]) if index == 2 else 0)
                method = "mismatch"
                bad_poses = bads

            else:
                ins, dele = edit_distance(read, reference[pos - (len(splits[0])
                 if index > 0 else 0): pos + l - (len(splits[2]) if index == 0 else 0)])

                if len(ins) <= err_thresh and len(ins) < min_errs:
                    min_errs = len(ins)
                    best_start_pos = pos - (len(splits[0]) if index > 0 else 0) - (len(splits[1]) if index == 2 else 0)
                    method = "InsDel"
                    bad_poses = [ins, dele]

    return min_errs, best_start_pos, method, bad_poses

def process_read(args):
    read, num, idx, reference, rk_dict, agg_val, err_thresh = args

    read = read.strip().replace('N','A')
    min1, p1, meth1, bpos1 = process(read, idx, reference, rk_dict, agg_val, err_thresh)
    min2, p2, meth2, bpos2 = process(reverse_complement(read), idx, reference, rk_dict, agg_val, err_thresh)

    if min1 < min2 and min1 < np.inf:
        with open(f"read_locs/{num}.pkl", "wb") as f:
            pickle.dump((min1, p1, meth1, len(read), bpos1), f)

    elif min2 < np.inf:
        with open(f"read_locs/{num}.pkl", "wb") as f:
            pickle.dump((min2, p2, meth2, len(read), bpos2), f)
