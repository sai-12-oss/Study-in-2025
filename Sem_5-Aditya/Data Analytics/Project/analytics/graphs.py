import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_distribution(save_path):
    df = pd.read_csv(f"{save_path}/processed_reads.csv")

    df = df[df['pos'].str.lower() != 'nil']  # Keep rows where 'pos' is not 'Nil'
    df['pos'] = df['pos'].astype(int)  # Convert 'pos' to integer
    
    # Plotting a histogram of the read start positions
    plt.figure(figsize=(12, 6))
    plt.hist(df['pos'], bins=100, color='skyblue', edgecolor='black')  # Adjust 'bins' as needed for granularity
    
    # Adding labels and title
    plt.xlabel('Position on Reference Sequence')
    plt.ylabel('Frequency of Read Start Positions')
    plt.title('Distribution of Read Start Positions on the Reference Sequence')
    plt.savefig(f"{save_path}/read_distribution.png")

def coverage(save_path, n):
    df = pd.read_csv(f"{save_path}/processed_reads.csv")

    df = df[df['num_errs']!= 'Nil']  # Keep rows where 'num_errs' is not 'Nil'
    coverage_array = np.zeros(n)

    for i in range(len(df)):
        start = int(df['pos'].iloc[i])
        length = int(df['len_read'].iloc[i])
        end = start + length
        coverage_array[start:end] += 1

    # Normalize the coverage array
    coverage_array = coverage_array

    # mean coverage
    mean_coverage = np.mean(coverage_array)

    # standard deviation of coverage
    std_coverage = np.std(coverage_array)

    plt.figure(figsize=(12, 6))
    plt.plot(coverage_array)
    plt.xlabel('Position on Reference Sequence')
    plt.ylabel('Coverage')
    plt.title('Coverage of the Reference Sequence')
    plt.savefig(f"{save_path}/coverage.png")

    sensititvity = 8
    coverage_threshold = mean_coverage + sensititvity * std_coverage

    exon_regions = []
    in_exon = False
    start = 0

    for i in range(len(coverage_array)):
        if coverage_array[i] > coverage_threshold:
            if not in_exon:
                start = i
                in_exon = True
        else:
            if in_exon:
                end = i-1
                if (end-start+1)>=10 and (end-start+1)<=500: # based on the biological knowledge, we can set the minimum length of the exon to be 50
                    exon_regions.append((start, end))
                in_exon = False

    if in_exon and (len(coverage_array)-start+1)>=50 and (len(coverage_array)-start+1)<=500:
        exon_regions.append((start, len(coverage_array)-1))

    exon_lengths = [end-start+1 for start, end in exon_regions]

    coverage_exons = np.zeros(n)

    for start, end in exon_regions:
        average_coverage = np.mean(coverage_array[start:end+1])
        coverage_exons[start:end+1] = average_coverage

    plt.figure(figsize=(12, 6))
    plt.plot(coverage_exons)
    plt.xlabel('Position on Reference Sequence')
    plt.ylabel('Coverage')
    plt.title('Coverage of the Exons Regions')
    plt.savefig(f"{save_path}/coverage_exons.png")

    # Bar plot of the exon lengths
    plt.figure(figsize=(12, 6))
    plt.hist(exon_lengths, bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Exon Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Exon Lengths')
    # draw the quartiles on the plot
    plt.axvline(np.percentile(exon_lengths, 25), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(exon_lengths, 50), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(exon_lengths, 75), color='blue', linestyle='dashed', linewidth=1)
    plt.savefig(f"{save_path}/exon_lengths.png")

def statistics(save_path):
    with open(f"{save_path}/processed_reads.csv", 'r') as file:
        lines = file.readlines()

    lines = lines[1:]  # Skip the header line

    mismatch_indices = []
    error_indices = []
    read_indices = []

    for i in range(3066720):
        r = lines[i].strip().split(',')
        if r[0] == 'Nil':
            continue

        pos = int(r[1])
        if pos < 0:
            pos += 151100561
        l = int(r[3])
        for i in range(pos, pos + l):
            if i < 151100561:
                read_indices.append(i)

        meth = r[2]
        string = r[4]
        if string == '':
            continue
        string = string.split(';')
        string = [int(i.strip()) for i in string]   
        if meth == 'mismatch':
            for i in string:
                if i+pos < 151100561:
                    mismatch_indices.append(i+pos)
                    error_indices.append(i+pos)
        else:
            for i in string:
                if i+pos < 151100561:
                    error_indices.append(i+pos)

    mismatch_indices = list(set(mismatch_indices))
    error_indices = list(set(error_indices))
    read_indices = list(set(read_indices))

    plt.hist(mismatch_indices, bins=10000, label='mismatch')
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mismatch Positions')
    plt.savefig(f"{save_path}/mismatch_positions.png")

    plt.hist(error_indices, bins=10000, label='error')
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Error positions in X chromosome')
    plt.savefig(f"{save_path}/error_positions.png")

    plt.hist(read_indices, bins=10000, label='read')
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Read positions in X chromosome')
    plt.savefig(f"{save_path}/read_positions.png")

    # plot hist_error_indices[i]/hist_read_indices[i] for i in range(1000)
    hist_error_indices, _ = np.histogram(error_indices, bins=10000)
    hist_read_indices, _ = np.histogram(read_indices, bins=10000)

    plt.plot(hist_error_indices/hist_read_indices)
    # Change size of the plot
    # width = 30, height = 10
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.xlabel('Position')
    plt.ylabel('Error Rate')
    plt.title('Error rate in X chromosome')
    plt.savefig(f"{save_path}/error_rate.png")

    





        
        
