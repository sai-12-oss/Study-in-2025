import sys
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


sys.setrecursionlimit(1000000)

# visualising a dendrogram using the community matrix
def visualise_dendrogram(mat, title, filepath):
    Z = sch.linkage(mat, method='ward')
    sch.dendrogram(Z)
    plt.xlabel("Nodes")
    plt.ylabel("Distance")
    plt.title(title)
    plt.savefig(filepath)