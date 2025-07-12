from utils import path_to_hyperedge, next_batch_index
import faiss
import torch
import numpy as np


def evaluation(city, query_traj_df, base_traj_df, emb_model, device):
    print(f"\n--- Trajectory Similarity Search ({city}) ---")
    query_vectors = []
    query_labels = []
    emb_model.eval()
    with torch.no_grad():
        for n, batch_index in enumerate(next_batch_index(query_traj_df.shape[0], bs=1000, shuffle=False)):
            batch_data = query_traj_df.iloc[batch_index]
            hyperedge_index, hyperedge_label = path_to_hyperedge(batch_data.copy(), label='query_label')
            hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.int64, device=device)
            hyperedge_label = torch.tensor(hyperedge_label, device=device)
            _, query_vecs = emb_model(emb_model.segment_attr_enc,
                                      emb_model.segment_vis_feat,
                                      emb_model.edge_index, hyperedge_index)
            query_vectors.append(query_vecs.detach().cpu())
            query_labels.append(hyperedge_label.detach().cpu())
    query_vectors = torch.cat(query_vectors, dim=0).numpy()
    query_labels = torch.cat(query_labels, dim=0).numpy()

    base_vectors = []
    emb_model.eval()
    with torch.no_grad():
        for n, batch_index in enumerate(next_batch_index(base_traj_df.shape[0], bs=1000, shuffle=False)):
            batch_data = base_traj_df.iloc[batch_index]
            hyperedge_index = path_to_hyperedge(batch_data.copy(), None)
            hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.int64, device=device)
            _, base_vecs = emb_model(emb_model.segment_attr_enc,
                                     emb_model.segment_vis_feat,
                                     emb_model.edge_index, hyperedge_index)
            base_vectors.append(base_vecs.detach().cpu())
    base_vectors = torch.cat(base_vectors, dim=0).numpy()

    index = faiss.IndexFlatL2(emb_model.params['tra_hidden_dim'])
    index.add(base_vectors)
    D, I = index.search(query_vectors, 1000)

    num_queries = 5000
    hit = 0
    rank_sum = 0
    no_hit = 0
    for i, r in enumerate(I):
        if query_labels[i] in r:
            rank = np.where(r == query_labels[i])[0][0] + 1  # 加1确保排名从1开始
            rank_sum += rank
            if query_labels[i] in r[:10]:
                hit += 1
        else:
            no_hit += 1
    print(f'Mean Rank: {rank_sum / num_queries}, HR@10: {hit / (num_queries - no_hit)}, No Hit: {no_hit}')