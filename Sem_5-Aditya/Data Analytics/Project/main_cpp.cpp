#include <set>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
using namespace std;

class SuccinctRank {
public:
    SuccinctRank(const string& bwt, int block_size = 32)
        : bwt(bwt), bwt_len(bwt.size()), block_size(block_size) {
        num_blocks = bwt_len / block_size + 1;
        rank_tables();
        count();
    }

    string bwt;
    int bwt_len;
    int block_size;
    int num_blocks;
    vector<int> bias;
    vector<vector<int>> rank_table;

    int rank(int i, char c) const {
        int block = i / block_size;
        int offset = i % block_size;

        int block_rank = rank_table[block][char_to_index(c)];
        int inside_rank = std::count(bwt.begin() + block * block_size, bwt.begin() + block * block_size + offset, c);

        return block_rank + inside_rank;
    }

    void rank_tables() {
        rank_table.resize(num_blocks, vector<int>(4, 0));
        for (int i = 1; i < num_blocks; ++i) {
            int start = (i - 1) * block_size;
            int end = min(i * block_size, bwt_len);
            for (int j = 0; j < 4; ++j) {
                rank_table[i][j] = rank_table[i - 1][j];
                rank_table[i][j] += std::count(bwt.begin() + start, bwt.begin() + end, "ACGT"[j]);
            }
        }
    }

    void count() {
        int count_A = rank(bwt_len, 'A') + (bwt.back() == 'A');
        int count_C = rank(bwt_len, 'C') + (bwt.back() == 'C');
        int count_G = rank(bwt_len, 'G') + (bwt.back() == 'G');

        vector<int> counts = {count_A, count_C, count_G};

        bias.resize(4);
        partial_sum(counts.begin(), counts.end(), bias.begin() + 1);
    }

    int char_to_index(char c) const {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default: return -1;
        }
    }
};


vector<int> find(const string& read, const SuccinctRank& rank_struct, const vector<int>& idx, const string& ref) {
    int len_subread = read.size() / 3;
    vector<string> subreads = {read.substr(0, len_subread), read.substr(len_subread, len_subread), read.substr(2 * len_subread)};
    vector<vector<int>> subread_match(3);

    for (int k = 0; k < 3; ++k) {
        string subread = subreads[k];
        int start = 0, end = rank_struct.bwt.size() - 1;

        for (auto it = subread.rbegin(); it != subread.rend(); ++it) {
            char char_ = *it;
            int bias = rank_struct.bias[rank_struct.char_to_index(char_)];

            start = rank_struct.rank(start, char_) + bias;
            end = rank_struct.rank(end, char_) - (char_ != rank_struct.bwt[end]) + bias;

            if (start > end) {
                start = -1;
                break;
            }
        }

        if (start != -1) {
            for (int i = start; i <= end; ++i) {
                subread_match[k].push_back(idx[i]);
            }
        }
    }

    set<int> valids;
    for (int i = 0; i < 3; ++i) {
        int bias = i * len_subread;
        for (int j : subread_match[i]) {
            int start = j - bias;
            if (start >= 0 && start + read.size() <= ref.size()) {
                int mismatch = 0;
                for (int l = 0; l < read.size(); ++l) {
                    if (ref[start + l] != read[l]) {
                        ++mismatch;
                    }
                    if (mismatch > 2) {
                        break;
                    }
                }
                if (mismatch <= 2) {
                    valids.insert(start);
                }
            }
        }
    }

    return vector<int>(valids.begin(), valids.end());
}

int match(const vector<int>& valids, const vector<pair<int, int>>& interval) {
    for (int start : valids) {
        for (int i = 0; i < 6; ++i) {
            if (start >= interval[i].first && start <= interval[i].second) {
                return i;
            }
        }
    }
    return -1;
}

vector<int> exon_match(const vector<string>& reads, const SuccinctRank& rank_struct, const vector<int>& idx, const string& ref, const vector<pair<int, int>>& red_int, const vector<pair<int, int>>& green_int) {
    vector<double> red_counts(6, 0), green_counts(6, 0);

    #pragma omp parallel
    {
        vector<double> local_red_counts(6, 0), local_green_counts(6, 0);

        #pragma omp for nowait
        for (int i = 0; i < reads.size(); ++i) {
            const string& read = reads[i];
            vector<int> valids = find(read, rank_struct, idx, ref);

            // if no valid positions found, try the reverse complement
            if (valids.empty()) {
                string rev_comp_read = read;
                reverse(rev_comp_read.begin(), rev_comp_read.end());
                for (char& c : rev_comp_read) {
                    switch (c) {
                        case 'A': c = 'T'; break;
                        case 'C': c = 'G'; break;
                        case 'G': c = 'C'; break;
                        case 'T': c = 'A'; break;
                    }
                }
                valids = find(rev_comp_read, rank_struct, idx, ref);
            }

            int red_pos = match(valids, red_int);
            int green_pos = match(valids, green_int);

            // if the read lies in both red and green intervals, count it as 0.5 for each
            // else count it as 1 for the interval it lies in
            if (red_pos != -1 && green_pos != -1) {
                local_red_counts[red_pos] += 0.5;
                local_green_counts[green_pos] += 0.5;                
            } else if (red_pos != -1 && green_pos == -1) {
                local_red_counts[red_pos] += 1;
            } else if (red_pos == -1 && green_pos != -1) {
                local_green_counts[green_pos] += 1;
            }
        }

        for (int i = 0; i < 6; ++i) {
            #pragma omp atomic
            red_counts[i] += local_red_counts[i];
            #pragma omp atomic
            green_counts[i] += local_green_counts[i];
        }
    }

    // print result
    cout << "Red counts: ";
    for (double i : red_counts) {
        cout << i << " ";
    }
    cout << endl;
    cout << "Green counts: ";
    for (double i : green_counts) {
        cout << i << " ";
    }
    cout << endl;

    vector<int> result;
    result.insert(result.end(), red_counts.begin(), red_counts.end());
    result.insert(result.end(), green_counts.begin(), green_counts.end());

    return result;
}


int main(int argc, char* argv[]) {
    auto start = chrono::high_resolution_clock::now();

    // take arguments for file paths
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <bwt_file> <map_file> <ref_file> <reads_file> " << endl;
        return 1;
    }

    string bwt_file = argv[1];
    string idx_file_path = argv[2];
    string ref_file_path = argv[3];
    string reads_file_path = argv[4];

    ifstream f(bwt_file);
    string bwt((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    f.close();
    bwt.erase(remove(bwt.begin(), bwt.end(), '\n'), bwt.end());

    ifstream idx_file(idx_file_path);
    vector<int> idx;
    string line;
    while (getline(idx_file, line)) {
        idx.push_back(stoi(line));
    }
    idx_file.close();

    ifstream ref_file(ref_file_path);
    getline(ref_file, line); // skip the first line
    string ref((istreambuf_iterator<char>(ref_file)), istreambuf_iterator<char>());
    ref.erase(remove(ref.begin(), ref.end(), '\n'), ref.end());
    ref_file.close();

    ifstream reads_file(reads_file_path);
    vector<string> reads;
    while (getline(reads_file, line)) {
        reads.push_back(line);
    }
    reads_file.close();

    cout << "Files read" << endl;

    vector<pair<int, int>> red_int = {{149249757, 149249868}, {149256127, 149256423}, {149258412, 149258580},
                                      {149260048, 149260213}, {149261768, 149262007}, {149264290, 149264400}};
    vector<pair<int, int>> green_int = {{149288166, 149288277}, {149293258, 149293554}, {149295542, 149295710},
                                        {149297178, 149297343}, {149298898, 149299137}, {149301420, 149301530}};

    SuccinctRank rank_struct(bwt);
    cout << "Rank structure created" << endl;

    vector<int> result = exon_match(reads, rank_struct, idx, ref, red_int, green_int);

    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::seconds>(end - start).count() << " seconds" << endl;

    return 0;
}