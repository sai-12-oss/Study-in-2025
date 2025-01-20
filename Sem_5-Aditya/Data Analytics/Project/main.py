import argparse
from preprocess.process_data import get_data
from align.align_reads import align_reads
from analytics.clean_data import cleanup
from analytics.graphs import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Matching')

    parser.add_argument('--bwt', type=str, help='BWT file path', default='gene_data/chrX_last_col.txt')
    parser.add_argument('--map', type=str, help='Character Map file path', default='gene_data/chrX_map.txt')
    parser.add_argument('--ref', type=str, help='Reference Sequence file path', default='gene_data/chrX.fa')
    parser.add_argument('--reads', type=str, help='Reads file path', default='gene_data/reads')
    parser.add_argument('--err_thresh', type=int, help='Error Threshold', default=2)
    parser.add_argument('--save_path', type=str, help='Save Path', default='results')

    args = parser.parse_args()

    bwt_path = args.bwt
    map_path = args.map
    fa_path = args.ref
    reads_path = args.reads
    err_thresh = args.err_thresh
    save_path = args.save_path

    print("Reading Data...")
    bwt, len_n, idx, reference, reads = get_data(bwt_path, map_path, fa_path, reads_path)
    print("Data Read Successfully!")
    print()

    print("Aligning Reads...")

    align_reads(bwt, len_n, idx, reference, reads, err_thresh)

    print("Reads Aligned Successfully!")
    print()

    print("Cleaning Data...")
    cleanup(save_path)
    print("Data Cleaned Successfully!")
    print()

    print("Output saved to processed_reads.csv")
    print()

    print("Creating Graphs...")
    read_distribution(save_path)
    coverage(save_path, len(reference))
    statistics(save_path)

    print("Graphs Created Successfully!")




    
