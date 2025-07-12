from layer import HieraGCN
from utils import node_hyper_relation, semi_loss
from utils import valid_node_edge_mask, hyper_edge_index_masking
from utils import next_batch_index, path_to_hyperedge, diag_to_zero

import time
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class UniTR(nn.Module):
    def __init__(self, params, road_network):
        super(UniTR, self).__init__()
        self.params = params
        self.road_network = road_network
        self.edge_index = from_scipy_sparse_matrix(self.road_network.LG_adj_sp)[0].to(self.params['device'])
        self.adjacency_matrix = torch.tensor(self.road_network.LG_adj,
                                             dtype=torch.float, device=self.params['device'])

        self.num_seg_ids = self.road_network.num_seg_ids
        self.num_seg_len_bins = self.road_network.num_seg_len_bins
        self.num_seg_lng_bins = self.road_network.num_seg_lng_bins
        self.num_seg_lat_bins = self.road_network.num_seg_lat_bins

        self.segment_attr_enc = torch.tensor(self.road_network.segment_attr_enc,
                                             dtype=torch.long, device=self.params['device'])

        self.segment_vis_feat = torch.tensor(self.road_network.segment_vis_feat,
                                             dtype=torch.float, device=self.params['device'])

        self.segment_id_embedding = nn.Embedding(self.num_seg_ids + 1, self.params['seg_id_dim'],
                                                 padding_idx=self.num_seg_ids)
        self.segment_len_embedding = nn.Embedding(self.num_seg_len_bins + 1, self.params['seg_len_dim'],
                                                  padding_idx=self.num_seg_len_bins)
        self.segment_lng_embedding = nn.Embedding(self.num_seg_lng_bins + 1, self.params['seg_lng_dim'],
                                                  padding_idx=self.num_seg_lng_bins)
        self.segment_lat_embedding = nn.Embedding(self.num_seg_lat_bins + 1, self.params['seg_lat_dim'],
                                                  padding_idx=self.num_seg_lat_bins)

        self.segment_attr_dim = self.params['seg_len_dim'] + self.params['seg_id_dim'] + \
                                self.params['seg_lng_dim'] + self.params['seg_lat_dim']
        self.segment_in_dim = self.segment_attr_dim + self.params['seg_vis_dim']

        self.graph_encoder = HieraGCN(in_dim=self.segment_in_dim,
                                      node_dim=self.params['seg_hidden_dim'],
                                      edge_dim=self.params['tra_hidden_dim'])

        self.road_network_projector = nn.Sequential(
            nn.Linear(self.params['seg_hidden_dim'], self.params['seg_latent_dim1']),
            nn.ELU(inplace=True),
            nn.Linear(self.params['seg_latent_dim1'], self.params['seg_latent_dim2'])
        )

        self.trajectory_projector = nn.Sequential(
            nn.Linear(self.params['tra_hidden_dim'], self.params['tra_latent_dim1']),
            nn.ELU(inplace=True),
            nn.Linear(self.params['tra_latent_dim1'], self.params['tra_latent_dim2'])
        )
        self.bi_linear = nn.Bilinear(self.params['seg_latent_dim2'], self.params['tra_latent_dim2'], 1)

        self.model_name = str(type(self).__name__)

    def forward(self, seg_attr, seg_vis_feat, edge_index, hyperedge_index, num_nodes=None, num_hyperedges=None):
        seg_id_feat = self.segment_id_embedding(seg_attr[:, 0])
        seg_len_feat = self.segment_len_embedding(seg_attr[:, 1])
        seg_lng_feat = self.segment_lng_embedding(seg_attr[:, 2])
        seg_lat_feat = self.segment_lat_embedding(seg_attr[:, 3])
        seg_feats = torch.cat([seg_id_feat, seg_len_feat, seg_lng_feat, seg_lat_feat, seg_vis_feat], dim=-1)
        embeddings = self.graph_encoder(x=seg_feats, edge_index=edge_index, hyperedge_index=hyperedge_index,
                                        full_num_nodes=num_nodes, full_num_hyperedges=num_hyperedges)
        return embeddings

    def segment_feature_masking(self, seg_attr, seg_vis_feat):
        mask_choices = torch.randint(3, (seg_attr.shape[0],))

        attr_mask = (mask_choices == 1)
        seg_attr[attr_mask] = torch.tensor([
            self.num_seg_ids,
            self.num_seg_len_bins,
            self.num_seg_lng_bins,
            self.num_seg_lat_bins
        ], device=seg_attr.device)

        vis_mask = (mask_choices == 2)
        seg_vis_feat[vis_mask] = 0.

        return seg_attr, seg_vis_feat

    def bi_linear_similarity(self, x1, x2, tau):
        sim = torch.sigmoid(self.bi_linear(x1, x2)).squeeze()
        sim = torch.exp(sim / tau)
        return sim

    def cross_contrastive_loss(self, z_seg, z_tra, hyper_edge_index, is_decouple=False):
        z_seg_perm = z_seg[torch.randperm(z_seg.size(0))]
        z_tra_perm = z_tra[torch.randperm(z_tra.size(0))]

        pos_sim = self.bi_linear_similarity(z_seg[hyper_edge_index[0]],
                                            z_tra[hyper_edge_index[1]], self.params['cross_tau'])
        seg_neg_sim = self.bi_linear_similarity(z_seg[hyper_edge_index[0]],
                                                z_tra_perm[hyper_edge_index[1]], self.params['cross_tau'])
        tra_neg_sim = self.bi_linear_similarity(z_seg_perm[hyper_edge_index[0]],
                                                z_tra[hyper_edge_index[1]], self.params['cross_tau'])

        if is_decouple:
            loss_seg = -torch.log(pos_sim / seg_neg_sim)
            loss_tra = -torch.log(pos_sim / tra_neg_sim)
        else:
            loss_seg = -torch.log(pos_sim / (pos_sim + seg_neg_sim))
            loss_tra = -torch.log(pos_sim / (pos_sim + tra_neg_sim))

        loss_seg = loss_seg[~torch.isnan(loss_seg)]
        loss_tra = loss_tra[~torch.isnan(loss_tra)]
        loss = (loss_seg + loss_tra).mean()
        return loss

    @staticmethod
    def same_contrastive_loss(z1, z2, tau, pos_mask, neg_mask, is_decouple=False):
        l1 = semi_loss(z1, z2, tau, pos_mask, neg_mask, is_decouple)
        l2 = semi_loss(z2, z1, tau, pos_mask, neg_mask, is_decouple)
        loss = ((l1 + l2) * 0.5).mean()
        return loss

    def save(self, epoch):
        prefix = './checkpoints/'
        file_marker = f"{self.model_name}_e{epoch}_b{self.params['num_hyperedges']}_{self.params['city_name']}"
        model_path = time.strftime(prefix + '%m%d_%H_%M_' + file_marker + '.pth')
        torch.save(self.state_dict(), model_path)
        print('save parameters to file: %s' % model_path)

    def load(self, filepath, device):
        self.load_state_dict(torch.load(filepath, map_location=device))
        print('Load parameters from file: %s' % filepath)

    def train_process(self, traj_df, is_decouple=False):
        print(f'Training RNTrajCL model on {self.params["city_name"]} dataset(DEVICE:{self.params["device"]}) ...')
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.params['lr'], weight_decay=self.params['wd'])

        self.train()
        best_loss = 1e9
        for e in range(1, self.params['num_epochs'] + 1):
            t0 = time.time()
            for n, batch_index in enumerate(next_batch_index(traj_df.shape[0],
                                                             self.params['num_hyperedges'], shuffle=True)):
                batch_data = traj_df.iloc[batch_index]
                hyperedge_index = path_to_hyperedge(batch_data.copy())
                hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.int64, device=self.params['device'])

                num_nodes = self.road_network.num_seg_ids
                num_hyperedges = int(hyperedge_index[1].max()) + 1

                '''augmentation'''
                aug_seg_attr1, aug_seg_vis1 = self.segment_feature_masking(self.segment_attr_enc.clone(),
                                                                           self.segment_vis_feat.clone())
                aug_seg_attr2, aug_seg_vis2 = self.segment_feature_masking(self.segment_attr_enc.clone(),
                                                                           self.segment_vis_feat.clone())
                aug_edge_index1, _ = dropout_edge(self.edge_index.clone(), p=self.params['edge_remove_ratio'])
                aug_edge_index2, _ = dropout_edge(self.edge_index.clone(), p=self.params['edge_remove_ratio'])

                aug_hyperedge_index1, _ = dropout_edge(hyperedge_index.clone(), p=self.params['hyper_remove_ratio'])
                aug_hyperedge_index2, _ = dropout_edge(hyperedge_index.clone(), p=self.params['hyper_remove_ratio'])

                '''hierarchical graph encoder'''
                seg_h1, tra_h1 = self.forward(aug_seg_attr1, aug_seg_vis1, aug_edge_index1, aug_hyperedge_index1,
                                              num_nodes, num_hyperedges)
                seg_h2, tra_h2 = self.forward(aug_seg_attr2, aug_seg_vis2, aug_edge_index2, aug_hyperedge_index2,
                                              num_nodes, num_hyperedges)

                '''projection'''
                seg_z1 = self.road_network_projector(seg_h1)
                seg_z2 = self.road_network_projector(seg_h2)

                tra_z1 = self.trajectory_projector(tra_h1)
                tra_z2 = self.trajectory_projector(tra_h2)

                '''contrast'''
                traj_count_bet_segs, jac_sim_bet_trajs = node_hyper_relation(hyperedge_index.clone(),
                                                                             num_nodes, num_hyperedges)

                seg_pos_mask = self.adjacency_matrix.clone()
                seg_pos_mask = diag_to_zero(seg_pos_mask)
                seg_neg_mask = (traj_count_bet_segs == 0.).float()

                _, hyperedge_valid_mask1 = valid_node_edge_mask(aug_hyperedge_index1, num_nodes, num_hyperedges)
                _, hyperedge_valid_mask2 = valid_node_edge_mask(aug_hyperedge_index2, num_nodes, num_hyperedges)
                hyperedge_valid_mask = hyperedge_valid_mask1 & hyperedge_valid_mask2

                jac_sim_bet_trajs = diag_to_zero(jac_sim_bet_trajs)
                tra_pos_mask = (jac_sim_bet_trajs >= 0.8).float()
                tra_pos_mask = tra_pos_mask[hyperedge_valid_mask, :][:, hyperedge_valid_mask]
                tra_neg_mask = (jac_sim_bet_trajs == 0.).float()
                tra_neg_mask = tra_neg_mask[hyperedge_valid_mask, :][:, hyperedge_valid_mask]

                masked_hyper_index1 = hyper_edge_index_masking(hyperedge_index.clone(), num_nodes,
                                                               num_hyperedges, None, hyperedge_valid_mask1)
                masked_hyper_index2 = hyper_edge_index_masking(hyperedge_index.clone(), num_nodes,
                                                               num_hyperedges, None, hyperedge_valid_mask2)

                loss_s = self.same_contrastive_loss(seg_z1, seg_z2, self.params['seg_tau'],
                                                    seg_pos_mask, seg_neg_mask, is_decouple)

                loss_t = self.same_contrastive_loss(z1=tra_z1[hyperedge_valid_mask],
                                                    z2=tra_z2[hyperedge_valid_mask],
                                                    tau=self.params['tra_tau'],
                                                    pos_mask=tra_pos_mask,
                                                    neg_mask=tra_neg_mask, is_decouple=is_decouple)

                loss_m = (self.cross_contrastive_loss(seg_z1, tra_z2, masked_hyper_index2) +
                          self.cross_contrastive_loss(seg_z2, tra_z1, masked_hyper_index1)) / 2

                loss = self.params['lambda1'] * loss_s + loss_t * self.params['lambda2'] + loss_m * self.params[
                    'lambda3']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (n + 1) % 50 == 0:
                    print(f'Epoch:{e}, Batch:{n + 1}, Loss:{loss.item():.5f}'
                          f'({loss_s.item():.5f}, {loss_t.item():.5f}, {loss_m.item():.5f}),'
                          f'Time:{round(time.time() - t0, 2)}')

            if loss < best_loss:
                best_loss = loss
                with torch.no_grad():
                    cur_seg_emb, _ = self.forward(self.segment_attr_enc,
                                                  self.segment_vis_feat,
                                                  self.edge_index, hyperedge_index)
                cur_seg_emb = cur_seg_emb.detach().cpu().numpy()
                np.savez_compressed(
                    f'checkpoints/segment_emb_{e}_{self.params["num_hyperedges"]}_{self.params["lambda1"]}{self.params["lambda2"]}{self.params["lambda3"]}',
                    data=cur_seg_emb)
                self.save(epoch=e)