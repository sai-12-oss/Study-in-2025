class SuccinctRank:
    def __init__(self, bit_string, char, large_block_size=32, small_block_size=4):
        self.bit_string = bit_string
        self.char = char
        self.large_block_size = large_block_size
        self.small_block_size = small_block_size
        self.num_large_blocks = (len(bit_string) + large_block_size - 1) // large_block_size
        self.large_blocks = [0] * self.num_large_blocks
        self.small_blocks = [[] for _ in range(self.num_large_blocks)]
        self._precompute_large_blocks()
        self._precompute_small_blocks()

    def _precompute_large_blocks(self):
        """Precompute cumulative ranks up to each large block."""
        cumulative_rank = 0
        for i in range(self.num_large_blocks):
            start = i * self.large_block_size
            end = min(start + self.large_block_size, len(self.bit_string))
            cumulative_rank += self.bit_string[start:end].count(self.char)
            self.large_blocks[i] = cumulative_rank

    def _precompute_small_blocks(self):
        """Precompute ranks within each large block at small block boundaries."""
        for i in range(self.num_large_blocks):
            start = i * self.large_block_size
            end = min(start + self.large_block_size, len(self.bit_string))
            cumulative_rank = 0
            small_block_ranks = []
            for j in range(start, end, self.small_block_size):
                small_end = min(j + self.small_block_size, end)
                cumulative_rank += self.bit_string[j:small_end].count(self.char)
                small_block_ranks.append(cumulative_rank)
            self.small_blocks[i] = small_block_ranks


    def rank(self, index):
        """Return the number of occurences in the bit string preceding the given index."""
        large_block_index = index // self.large_block_size
        small_block_index = (index % self.large_block_size) // self.small_block_size

        rank = self.large_blocks[large_block_index - 1] if large_block_index > 0 else 0

        if small_block_index > 0:
            rank += self.small_blocks[large_block_index][small_block_index - 1]

        small_block_start = large_block_index * self.large_block_size + small_block_index * self.small_block_size

        for i in range(small_block_start, index):
            rank += self.bit_string[i] == self.char

        return rank