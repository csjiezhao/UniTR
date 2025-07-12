import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class RoadLabelClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(RoadLabelClassifier, self).__init__()
        self.net = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        :param x: (num_segments, embed_dim)
        :return:
        """
        return self.net(x)


def evaluation(city, emb_model, label_path, hyperedge_index, num_nodes, num_hyperedges,
               num_fold, epochs, num_classes, device):
    print(f"--- Road Type Classification ({city})---")
    emb_model.eval()
    with torch.no_grad():
        embeddings, _ = emb_model(emb_model.segment_attr_enc,
                                  emb_model.segment_vis_feat,
                                  emb_model.edge_index, hyperedge_index, num_nodes, num_hyperedges)
        embeddings = embeddings.detach()
    labels = np.load(label_path)['data']
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    valid_indices = labels != -1
    embeddings = embeddings[valid_indices]
    labels = labels[valid_indices]

    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    scores = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        task_model = RoadLabelClassifier(embed_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(task_model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_acc = 0.
        best_pred = None
        for e in range(1, epochs + 1):
            task_model.train()
            optimizer.zero_grad()
            pred_train = task_model(x_train)
            loss = criterion(pred_train, y_train)
            loss.backward()
            optimizer.step()

            task_model.eval()
            with torch.no_grad():
                pred_score = task_model(x_val)
                pred_val = torch.argmax(pred_score, -1).detach().cpu()
                acc = accuracy_score(y_val.detach().cpu(), pred_val)
                if acc > best_acc:
                    best_acc = acc
                    best_pred = pred_val
                    best_score = torch.softmax(pred_score, dim=1)

        preds.append(best_pred)
        scores.append(best_score.detach().cpu())
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    scores = torch.cat(scores, dim=0)
    trues = torch.cat(trues, dim=0)
    macro_f1 = f1_score(trues, preds, average='macro')
    micro_f1 = f1_score(trues, preds, average='micro')
    print(f'Mi-F1: {micro_f1:.4f}, Ma-F1: {macro_f1:.4f}')
    return micro_f1, macro_f1


def evaluation_with_emb(city, emb_path, label_path, num_fold, epochs, num_classes, device):
    print(f"--- Road Type Classification ({city})---")
    embeddings = np.load(emb_path)['data']
    embeddings = torch.tensor(embeddings, dtype=torch.float, device=device)
    labels = np.load(label_path)['data']
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    valid_indices = labels != -1
    embeddings = embeddings[valid_indices]
    labels = labels[valid_indices]

    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    scores = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        task_model = RoadLabelClassifier(embed_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(task_model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_acc = 0.
        best_pred = None
        for e in range(1, epochs + 1):
            task_model.train()
            optimizer.zero_grad()
            pred_train = task_model(x_train)
            loss = criterion(pred_train, y_train)
            loss.backward()
            optimizer.step()

            task_model.eval()
            with torch.no_grad():
                pred_score = task_model(x_val)
                pred_val = torch.argmax(pred_score, -1).detach().cpu()
                acc = accuracy_score(y_val.detach().cpu(), pred_val)
                if acc > best_acc:
                    best_acc = acc
                    best_pred = pred_val
                    best_score = torch.softmax(pred_score, dim=1)

        preds.append(best_pred)
        scores.append(best_score.detach().cpu())
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    scores = torch.cat(scores, dim=0)
    trues = torch.cat(trues, dim=0)
    macro_f1 = f1_score(trues, preds, average='macro')
    micro_f1 = f1_score(trues, preds, average='micro')
    print(f'Mi-F1: {micro_f1:.4f}, Ma-F1: {macro_f1:.4f}')
    return micro_f1, macro_f1