from utils import path_to_hyperedge, next_batch_index

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader


class MLPReg(nn.Module):
    def __init__(self, input_size, num_layers, activation):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


def evaluation(city, traj_df, emb_model, epochs, device):
    print(f"\n--- Travel Time Estimation ({city}) ---")
    emb_model.eval()
    embeddings = []
    labels = []
    for n, batch_index in enumerate(next_batch_index(traj_df.shape[0], bs=1000, shuffle=False)):
        batch_data = traj_df.iloc[batch_index]
        hyperedge_index, hyperedge_tt = path_to_hyperedge(batch_data.copy(), label='travel_time')
        hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.int64, device=device)
        hyperedge_tt = torch.tensor(hyperedge_tt, dtype=torch.float, device=device).squeeze(-1)

        _, hyperedge_emb = emb_model(emb_model.segment_attr_enc,
                                     emb_model.segment_vis_feat,
                                     emb_model.edge_index, hyperedge_index)
        embeddings.append(hyperedge_emb.detach())
        labels.append(hyperedge_tt)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    batch_size = 64
    task_model = MLPReg(emb_model.params['tra_hidden_dim'], 3, nn.ReLU()).to(device)
    optimizer = torch.optim.Adam(task_model.parameters())
    criterion = nn.MSELoss().to(device)

    split = int(embeddings.shape[0] * 0.8)
    x_train, x_test = embeddings[:split], embeddings[split]
    y_train, y_test = labels[:split], labels[split]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)

    best = [0, 1e9, 1e9]  # best epoch, best mae, best rmse
    for e in range(1, epochs + 1):
        task_model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            y_pred = task_model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        task_model.eval()
        trues = []
        preds = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                trues.append(y.cpu())
                preds.append(task_model(x).cpu())
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        mae = mean_absolute_error(trues, preds)
        rmse = mean_squared_error(trues, preds) ** 0.5
        if e % 20 == 0:
            print(f'Epoch: {e}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        if mae < best[1]:
            best = [e, mae, rmse]
    print(f'Best epoch: {best[0]}, MAE: {best[1]:.4f}, RMSE: {best[2]:.4f}')
    return best[1], best[2]
