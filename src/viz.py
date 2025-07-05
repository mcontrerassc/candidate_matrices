import matplotlib.pyplot as plt
import numpy as np
from src.markov import forward_convert, backward_convert

def old_show_matrix(M, title=None, labels = None, cmap='viridis'):
    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    img = ax.imshow(M, cmap=cmap)
    fig.colorbar(img)
    if labels:
        ax.set_xticks(range(M.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(M.shape[1]))
        ax.set_yticklabels(labels)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_aspect('auto')
    plt.show()

def show_matrix(M, title=None, labels=None, cmap='viridis', boundaries=None):
    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    img = ax.imshow(M, cmap=cmap)
    fig.colorbar(img)
    if labels:
        ax.set_xticks(range(M.shape[1]), minor=False)
        ax.set_yticks(range(M.shape[0]), minor=False)
        ax.grid(False)
        #ax.set_xticks(range(M.shape[0]))
        ax.set_xticklabels(labels)
        #ax.set_yticks(range(M.shape[1]))
        ax.set_yticklabels(labels)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_aspect('auto')

    # draw boundary lines
    if boundaries.any():
        for b in boundaries:
            ax.axhline(b - 0.5, color='black', linewidth=1)
            ax.axvline(b - 0.5, color='black', linewidth=1)

    plt.show()

def viz_partition(partition, boost, candidates):
    if type(partition) == np.ndarray:
        partish = backward_convert(partition, candidates)
    else:
        partish = partition.copy()
    ordering = [c for bloc in partish for c in bloc]
    permutation_list = []
    for candidate in candidates:
        permutation_list.append(
            ordering.index(candidate)
        )
    permutation_matrix = np.zeros((len(candidates), len(candidates)))
    for i, p in enumerate(permutation_list):
        permutation_matrix[i, p] = 1

    # determine boundaries between blocks
    block_sizes = [len(b) for b in partish]
    boundaries = np.cumsum(block_sizes)[:-1]  # omit final edge

    show_matrix(
        permutation_matrix.T @ boost @ permutation_matrix,
        labels=ordering,
        cmap='PRGn',
        boundaries=boundaries
    )

def old_viz_order(cands,partition,boost):
    ordering = [c for bloc in partition for c in bloc]
    print(len(ordering), len(cands))
    permutation_list = []
    for candidate in cands:
        permutation_list.append(
            ordering.index(candidate)
        )

    permutation_matrix = np.zeros((len(cands),len(cands)))
    for i, p in enumerate(permutation_list):
        permutation_matrix[i,p] = 1
    show_matrix(permutation_matrix.T @ boost @ permutation_matrix, labels = ordering, cmap='PRGn')