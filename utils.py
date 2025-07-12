import torch
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_scatter import scatter_add
import swifter
import ast


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_loss_curve(seq1, name1, seq2, name2):
    fig, ax = plt.subplots()
    x = np.arange(1, len(seq1) + 1)
    ax.plot(x, seq1, label=name1)
    ax.plot(x, seq2, label=name2)
    ax.legend()
    ax.set_xlabel('Num of Epochs')
    ax.set_ylabel('Value of Loss')
    plt.show()


def cosine_similarity(x1, x2, tau):
    x1 = F.normalize(x1)
    x2 = F.normalize(x2)
    sim = torch.matmul(x1, x2.t())
    sim = torch.exp(sim / tau)
    return sim


def semi_loss(z1, z2, tau, pos_mask, neg_mask, is_decouple=False):
    intra_sim = cosine_similarity(z1, z1, tau)
    inter_sim = cosine_similarity(z1, z2, tau)
    intra_pos = (intra_sim * pos_mask).sum(1)
    inter_pos = inter_sim.diag()
    intra_neg = (intra_sim * neg_mask).sum(1)
    inter_neg = (inter_sim * neg_mask).sum(1)

    if is_decouple:
        return -torch.log(
            (inter_pos + intra_pos)
            / (inter_neg + intra_neg))
    else:
        return -torch.log(
            (inter_pos + intra_pos)
            / (inter_pos + intra_pos + inter_neg + intra_neg))


def node_hyper_relation(hyperedge_index, num_nodes, num_hyperedges):
    nodes = hyperedge_index[0]
    hyperedges = hyperedge_index[1]

    incidence_matrix = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32, device=hyperedge_index.device)
    incidence_matrix[nodes, hyperedges] = 1

    hyper_occur_bet_nodes = torch.mm(incidence_matrix, incidence_matrix.t())  # num_nodes, num_nodes
    node_occur_bet_hypers = torch.mm(incidence_matrix.t(), incidence_matrix)

    hyperedge_sizes = incidence_matrix.sum(dim=0)
    size_matrix_row = hyperedge_sizes.view(-1, 1)
    size_matrix_col = hyperedge_sizes.view(1, -1)
    union_matrix = size_matrix_row + size_matrix_col - node_occur_bet_hypers
    hyper_jaccard_sim = node_occur_bet_hypers / union_matrix
    hyper_jaccard_sim[union_matrix == 0] = 0

    return hyper_occur_bet_nodes, hyper_jaccard_sim


def valid_node_edge_mask(hyper_edge_index, num_nodes, num_edges):
    ones = hyper_edge_index.new_ones(hyper_edge_index.shape[1], device=hyper_edge_index.device)
    Dn = scatter_add(ones, hyper_edge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyper_edge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def hyper_edge_index_masking(hyper_edge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyper_edge_index
    H = torch.sparse_coo_tensor(indices=hyper_edge_index, values=hyper_edge_index.
                                new_ones((hyper_edge_index.shape[1],)), size=(num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyper_edge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyper_edge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyper_edge_index = H[node_mask].to_sparse().indices()
    return masked_hyper_edge_index


def path_to_hyperedge(df, label=None):
    df.loc[:, 'hyperedge_id'] = np.arange(df.shape[0])
    df1 = df[['path', 'hyperedge_id']]
    df1 = df1.explode('path')
    hyperedge_index = df1.to_numpy().T
    if label is not None:
        hyperedge_label = df[[label]].values
        return hyperedge_index.astype(float), hyperedge_label.astype(float)
    else:
        return hyperedge_index.astype(float)


def get_traj_train_set(traj_files, num_samples, seed):
    min_len, max_len = 10, 100
    dfs = []
    for file in traj_files:
        df = pd.read_csv(file, usecols=['path', 'path_len'])
        df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]
        dfs.append(df[['path']])
    df = pd.concat(dfs).reset_index(drop=True)
    if len(df) > num_samples:
        df = df.sample(n=num_samples, replace=False, random_state=seed)
    df['path'] = df['path'].swifter.apply(ast.literal_eval)
    print("Training trajectory load done...")
    return df


def get_traj_tte_set(traj_files):
    min_len, max_len = 1, 100
    dfs = []
    for file in traj_files:
        df = pd.read_csv(file, index_col=None, usecols=['path', 'path_len', 'travel_time'])
        df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    df['path'] = df['path'].swifter.apply(ast.literal_eval)
    return df


def get_traj_srh_set(traj_files, padding_idx, seed):
    num_queries = 5000
    db_size = 100000
    detour_rate = 0.1

    min_len, max_len = 10, 100
    dfs = []
    for file in traj_files:
        df = pd.read_csv(file, index_col=None, usecols=['path', 'path_len'])
        df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    if len(df) > db_size:
        df = df.sample(n=db_size, replace=False, random_state=seed)
    # df = df.reset_index(drop=True)

    df['path'] = df['path'].swifter.apply(ast.literal_eval)

    def detour():
        return np.random.randint(padding_idx)

    query_tra = []
    query_len = []
    random_idx = np.random.permutation(db_size)
    for i in range(num_queries):
        row = df.iloc[random_idx[i]]
        detour_pos = np.random.choice(row['path_len'], int(row['path_len'] * detour_rate), replace=False)
        path = [e for i, e in enumerate(row['path']) if i not in detour_pos]
        # path = [detour() if i in detour_pos else e for i, e in enumerate(row['path'])]
        query_tra.append(path)
        query_len.append(row['path_len'])
    query_label = random_idx[:num_queries]

    query_df = pd.DataFrame({'path': query_tra, 'path_len': query_len, 'query_label': query_label})
    return query_df, df


def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)
    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)
    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index


def diag_to_zero(mat):
    diag = torch.diag_embed(torch.diag(mat))
    mat = mat - diag
    return mat


def random_replace_nodes(hyperedge_index, num_nodes, replace_prob=0.1):
    replace_mask = torch.rand(hyperedge_index.size(1), device=hyperedge_index.device) < replace_prob
    new_nodes = torch.randint(0, num_nodes, (hyperedge_index.size(1),), device=hyperedge_index.device)
    hyperedge_index[0] = torch.where(replace_mask, new_nodes, hyperedge_index[0])
    return hyperedge_index