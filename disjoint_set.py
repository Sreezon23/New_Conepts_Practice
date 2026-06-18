class DisjointSet:
    def __init__(self, size):
        self.parent = []
        self.rank = []
        i = 0
        while i < size:
            self.parent.append(i)
            self.rank.append(0)
            i += 1

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

if __name__ == "__main__":
    ds = DisjointSet(5)
    ds.union(0, 2)
    ds.union(4, 2)
    ds.union(3, 1)
    print(ds.find(4) == ds.find(0))
    print(ds.find(1) == ds.find(0))
