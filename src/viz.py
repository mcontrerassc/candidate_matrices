import matplotlib.pyplot as plt
import numpy as np
from src.markov import forward_convert, backward_convert

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

def viz_partition(partition, boost, candidates, cmap='PRGn'):
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
        cmap=cmap,
        boundaries=boundaries
    )

def viz_best_partitions(
    jsonl_path,
    boost,
    candidates,
    cmap='PRGn'
):
    # load partitions
    partitions = []
    with open(jsonl_path, "r") as f:
        for line in f:
            arr = np.array(json.loads(line), dtype=np.int8)
            partitions.append(arr)
    
    num_parts = len(partitions)
    if num_parts == 0:
        print("No partitions found in file.")
        return
    
    fig, axes = plt.subplots(
        1, num_parts, figsize=(5 * num_parts, 5)
    )
    
    # if only one axis, make it a list to treat uniformly
    if num_parts == 1:
        axes = [axes]
    
    for ax, partition in zip(axes, partitions):
        if isinstance(partition, np.ndarray):
            partish = backward_convert(partition, candidates)
        else:
            partish = partition.copy()
        
        ordering = [c for bloc in partish for c in bloc]
        permutation_list = []
        for candidate in candidates:
            permutation_list.append(ordering.index(candidate))
        
        permutation_matrix = np.zeros((len(candidates), len(candidates)))
        for i, p in enumerate(permutation_list):
            permutation_matrix[i, p] = 1
        
        block_sizes = [len(b) for b in partish]
        boundaries = np.cumsum(block_sizes)[:-1]
        
        M = permutation_matrix.T @ boost @ permutation_matrix
        
        img = ax.imshow(M, cmap=cmap)
        fig.colorbar(img, ax=ax)
        
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_xticklabels(ordering, rotation=90)
        ax.set_yticklabels(ordering)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_aspect('auto')
        
        for b in boundaries:
            ax.axhline(b - 0.5, color='black', linewidth=1)
            ax.axvline(b - 0.5, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.show()