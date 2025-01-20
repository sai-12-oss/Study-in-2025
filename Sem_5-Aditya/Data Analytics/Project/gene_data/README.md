# Getting Gene Data

## Download X Chromosome Data

You can download the X chromosome data from the following link:

[Download X Chromosome Data](https://drive.google.com/file/d/1_DLln6OJOlwrXdJEQwwlIDhNuoRFDCxw/view?usp=sharing)

The zip file contains the following files:

1. **chrX.fa**: The reference sequence for Chr X in fasta format. The first line is the header line followed by the sequence (ignore the newlines).

2. **chrX_last_col.txt**: The last column of the BWT. This sequence contains one more character than the reference sequence as it also contains the special character `$` which is appended to its end.

3. **chrX_map.txt**: Contains mapping of indexes in BWT with indexes in the reference. Line number `i` (0-based) has the starting index in the reference of the `i`th sorted circular shift. For example, if the first line contains `3435`, it means that the string starting at `3435` (0-based) is the first in the sort order of BWT rotated strings.

4. **reads**: Contains about 3 million reads, one read per line. Reads are roughly of length 100, but may not be exactly so. Each read could come from the reference sequence or its reverse complement, so consider reverse complements of each read as well.

5. **Red and Green gene locations**: Each of these genes should begin with `ATGGCCCAGCAGTGGAGCCTC`. You can grep for it and the red and green starting positions should appear at (0-based):

    **Red exons:**
    - 149249757 - 149249868
    - 149256127 - 149256423
    - 149258412 - 149258580
    - 149260048 - 149260213
    - 149261768 - 149262007
    - 149264290 - 149264400

    **Green exons:**
    - 149288166 - 149288277
    - 149293258 - 149293554
    - 149295542 - 149295710
    - 149297178 - 149297343
    - 149298898 - 149299137
    - 149301420 - 149301530

As a test, use the following read:
`GAGGACAGCACCCAGTCCAGCATCTTCACCTACACCAACAGCAACTCCACCAGAGGTGAGCCAGCAGGCCCGTGGAGGCTGGGTGGCTGCACTGGGGGCCA`
which should match at `149249814`.
