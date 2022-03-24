class ReplayBuffer():
    def __init__(self, max_size):
        self.q = [None] * max_size
        self.max_size = max_size
        self.size = 0
        self.ptr = 0
        self.command_counts = [0] * 6

    def append(self, sample):
        self.q[self.ptr] = sample
        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size

    def to_list(self):
        return self.q[:self.size]

    def load_data(self, data_q, step):
        self.q = data_q[:self.max_size]
        self.size = min(step, self.max_size)
        self.ptr = step % self.max_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.q[idx]
