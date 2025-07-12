import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RoadLabelPredictor(nn.Module):
    def __init__(self, emb_dim):
        super(RoadLabelPredictor, self).__init__()
        self.net = nn.Linear(emb_dim, 1)

    def forward(self, x):
        return self.net(x)


def evaluation(city, emb_model, label_path, hyperedge_index, num_nodes, num_hyperedges,
               num_fold, epochs, device):
    print(f"--- Road Speed Inference ({city}) ---")
    emb_model.eval()
    with torch.no_grad():
        embeddings, _ = emb_model(emb_model.segment_attr_enc,
                                  emb_model.segment_vis_feat,
                                  emb_model.edge_index, hyperedge_index, num_nodes, num_hyperedges)
        embeddings = embeddings.detach()
    labels = np.load(label_path)['data']
    labels = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(-1)
    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        task_model = RoadLabelPredictor(embed_dim).to(device)
        optimizer = torch.optim.Adam(task_model.parameters())
        criterion = nn.MSELoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_mse = 1e9
        best_pred = None
        for e in range(1, epochs + 1):
            task_model.train()
            pred_train = task_model(x_train)
            loss = criterion(pred_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_model.eval()
            with torch.no_grad():
                pred_val = task_model(x_val).detach().cpu()
                mse = mean_squared_error(y_val.detach().cpu(), pred_val)
                if mse < best_mse:
                    best_mse = mse
                    best_pred = pred_val
        preds.append(best_pred)
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds) ** 0.5
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    return mae, rmse


def evaluation_with_emb(city, emb_path, label_path, num_fold, epochs, device):
    print(f"--- Road Speed Inference ({city}) ---")
    embeddings = np.load(emb_path)['data']
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    labels = np.load(label_path)['data']
    labels = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(-1)
    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        task_model = RoadLabelPredictor(embed_dim).to(device)
        optimizer = torch.optim.Adam(task_model.parameters())
        criterion = nn.MSELoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_mse = 1e9
        best_pred = None
        for e in range(1, epochs + 1):
            task_model.train()
            pred_train = task_model(x_train)
            loss = criterion(pred_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_model.eval()
            with torch.no_grad():
                pred_val = task_model(x_val).detach().cpu()
                mse = mean_squared_error(y_val.detach().cpu(), pred_val)
                if mse < best_mse:
                    best_mse = mse
                    best_pred = pred_val
        preds.append(best_pred)
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds) ** 0.5
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    return mae, rmse