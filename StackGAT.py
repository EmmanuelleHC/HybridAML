
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,roc_auc_score
import torch_geometric.nn as pyg_nn
from itertools import product
from torch_geometric.nn import GATConv
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import joblib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_paths = {
    "train": {"raw": "Thesis/train.csv", "processed": "Thesis/processed_train_gat.csv"},
    "val": {"raw": "Thesis/val.csv", "processed": "Thesis/processed_val_gat.csv"},
    "test": {"raw": "Thesis/test.csv", "processed": "Thesis/processed_test_gat.csv"},
}
scaler = StandardScaler()

def downsample_majority(df, target_column="Label", majority_class=0):
    majority_df = df[df[target_column] == majority_class]
    minority_df = df[df[target_column] == 1]
    downsampled_majority = majority_df.sample(len(minority_df), random_state=42)
    return pd.concat([downsampled_majority, minority_df], ignore_index=True)

def preprocess_and_load_data(raw_path, processed_path, scaler=None, downsample=False, is_train=False):
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        df = pd.read_csv(raw_path)
        df['Sender_Anonymous'] = (df['Sender_Account'] == "Not Provided").astype(int)

        df["Sender_Account"] = np.where(df["Sender_Account"] == "Not Provided", df["Bene_Account"], df["Sender_Account"])
        df["Bene_Id"] = np.where(df["Bene_Id"] == "Not Provided", df["Sender_Id"], df["Bene_Id"])
        df["Sender_Id"] = np.where(df["Sender_Id"] == "Not Provided", df["Bene_Id"], df["Sender_Id"])
        df["Sender_Institution"] = np.where(df["Sender_Institution"] == "Not Provided", df["Bene_Institution"], df["Sender_Institution"])
        df["Bene_Institution"] = np.where(df["Bene_Institution"] == "Not Provided", df["Sender_Institution"], df["Bene_Institution"])
        df["Bene_Account"] = np.where(df["Bene_Account"] == "Not Provided", df["Sender_Account"], df["Bene_Account"])
        df["Bene_Country"] = np.where(df["Bene_Country"] == "Not Provided", df["Sender_Country"], df["Bene_Country"])
        df["Sender_Country"] = np.where(df["Sender_Country"] == "Not Provided", df["Bene_Country"], df["Sender_Country"])
        
        if downsample:
            df = downsample_majority(df, target_column="Label")
        
    df["Country_Risk"] = df["Sender_Country"].apply(lambda x: 1 if x in ["Iran", "Syria", "North-Korea", "South Africa", "Panama"] else 0)
    df["Cross_Border"] = (df["Sender_Country"] != df["Bene_Country"]).astype(int)
    df["PEP_Involvement"] = (df["Sender_Is_Pep"] | df["Bene_Is_Pep"]).astype(int)
    if "USD_amount" in df.columns:
        if is_train:
            df["Scaled_USD_Amount"] = scaler.fit_transform(df[["USD_amount"]])
            joblib.dump(scaler, "hybrid_gat_scaler.pkl")
        else:
            if scaler is None:
                scaler = joblib.load("hybrid_gat_scaler.pkl")
            df["Scaled_USD_Amount"] = scaler.transform(df[["USD_amount"]])
    else:
        raise KeyError("USD_amount column is missing in the DataFrame.")


    df.to_csv(processed_path, index=False)

    return (df, scaler) if is_train else df
def process_datasets(data_paths, scaler):
    datasets = {}
    for split in ["train", "val", "test"]:
        raw_path = data_paths[split]["raw"]
        processed_path = data_paths[split]["processed"]
        downsample = (split == "train")
        is_train = (split == "train")
        
        if is_train:
            df, scaler = preprocess_and_load_data(
                raw_path, processed_path, scaler=scaler, downsample=True, is_train=True
            )
        else:
            df = preprocess_and_load_data(
                raw_path, processed_path, scaler=scaler, downsample=False, is_train=False
            )
        
        datasets[split] = {"df": df}

    return datasets
datasets = process_datasets(data_paths, scaler)
train_df = datasets["train"]["df"]
val_df = datasets["val"]["df"]
test_df = datasets["test"]["df"]
class AdaptiveGaussianMF(nn.Module):
    def __init__(self, mean, sigma):
        super(AdaptiveGaussianMF, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        return torch.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2))
class MemFuncs(nn.Module):
    def __init__(self, X):
        super(MemFuncs, self).__init__()
        self.MFList = nn.ModuleList(self.create_mem_funcs(X))

    def create_mem_funcs(self, X):
        MFList = []
        for feature_idx in range(X.shape[1]): 
            feature_data = X[:, feature_idx]
            mean_value = feature_data.mean().item()
            std_value = feature_data.std().item()
            MFList.append(
                nn.ModuleList([
                    AdaptiveGaussianMF(mean=mean_value - std_value, sigma=std_value),
                    AdaptiveGaussianMF(mean=mean_value, sigma=std_value),
                    AdaptiveGaussianMF(mean=mean_value + std_value, sigma=std_value)
                ])
            )
        return MFList

    def forward(self, X):
        layerOne = []
        for feature_idx, mfs in enumerate(self.MFList):
            input_col = X[:, feature_idx].unsqueeze(1) 
            col_result = [mf(input_col) for mf in mfs]  
            layerOne.append(torch.cat(col_result, dim=1)) 
        return torch.stack(layerOne, dim=1)  

class ANFIS(nn.Module):
    def __init__(self, X, Y, memFuncs):
        super(ANFIS, self).__init__()
        self.X = X.clone().detach().to(device)
        self.Y = Y.clone().detach().unsqueeze(1).to(device)
        self.memFuncs = memFuncs
        
        self.rules = torch.tensor(
            list(product(*[range(len(mf)) for mf in memFuncs.MFList])),
            dtype=torch.long,
        ).to(device)


        if self.rules.size(1) != X.shape[1]:
            raise ValueError(
                f"Rules tensor shape mismatch: expected {X.shape[1]} features, "
                f"but got rules tensor with shape {self.rules.shape}"
            )
        
        self.num_rules = len(self.rules)
        self.consequents = nn.Parameter(
            torch.randn(self.num_rules, X.shape[1] + 1) * 0.01
        )

    def forward(self, X):
        layerOne = self.memFuncs(X)
        batch_size = X.size(0)
        num_rules = self.rules.size(0)
        miAlloc = torch.ones(batch_size, num_rules).to(device)
        
        for feature_idx in range(X.shape[1]):
            rule_indices = self.rules[:, feature_idx].unsqueeze(0).expand(batch_size, -1)
            mf_values = torch.gather(layerOne[:, feature_idx, :], 1, rule_indices)
            miAlloc *= mf_values
        
        wSum = torch.sum(miAlloc, dim=1, keepdim=True) + 1e-10
        normalized = miAlloc / wSum
        extended_X = torch.cat([X, torch.ones(X.size(0), 1).to(device)], dim=1)
        weighted_consequents = torch.bmm(
            normalized.unsqueeze(1),
            self.consequents.unsqueeze(0).expand(X.size(0), -1, -1),
        )[:, 0, :]
        return torch.sum(weighted_consequents, dim=1)

    def train_step(self, optimizer):

        optimizer.zero_grad()
        Y_pred = self.forward(self.X)
        if Y_pred.dim() == 1:
            Y_pred = Y_pred.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(Y_pred, self.Y)
        loss.backward()
        optimizer.step()
        return loss.item()

memFuncs = MemFuncs(
    torch.tensor(train_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32)
).to(device)

X = torch.tensor(
    train_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values,
    dtype=torch.float32
).to(device)

Y = torch.tensor(train_df["Label"].values, dtype=torch.float32).to(device)

anfis = ANFIS(X, Y, memFuncs).to(device)


anfis_optimizer = Adam(anfis.parameters(), lr=0.01)

print("Training ANFIS...")
for epoch in range(20):
    anfis_loss = anfis.train_step(anfis_optimizer)
    print(f"Epoch {epoch + 1}, ANFIS Loss: {anfis_loss:.6f}")
anfis_model_path = "gat_anfis_model.pth"

def save_anfis_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"ANFIS model saved to {file_path}")
save_anfis_model(anfis, anfis_model_path)

anfis.eval()
with torch.no_grad():
    test_X = torch.tensor(
        test_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values,
        dtype=torch.float32
    ).to(device)
    test_predictions = anfis(test_X)
    test_predictions = torch.sigmoid(test_predictions)  
    test_predictions = (test_predictions > 0.5).float().cpu() 
test_y_tensor = torch.tensor(test_df["Label"].values, dtype=torch.float32).cpu()  
f1 = f1_score(test_y_tensor, test_predictions)
precision = precision_score(test_y_tensor, test_predictions)
recall = recall_score(test_y_tensor, test_predictions)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
train_scores = anfis.forward(
    torch.tensor(train_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32).to(device)
).detach().cpu().numpy()

val_scores = anfis.forward(
    torch.tensor(val_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32).to(device)
).detach().cpu().numpy()

test_scores = anfis.forward(
    torch.tensor(test_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32).to(device)
).detach().cpu().numpy()

train_df['Suspicious_Score'] = train_scores
val_df['Suspicious_Score'] = val_scores
test_df['Suspicious_Score'] = test_scores
def map_nodes_global(df_list):
    all_nodes = set()
    for df in df_list:
        all_nodes.update(df['Sender_Id'])
        all_nodes.update(df['Bene_Id'])
    global_mapping = {node: idx for idx, node in enumerate(all_nodes)}
    for df in df_list:
        df['Sender_Id_Mapped'] = df['Sender_Id'].map(global_mapping).fillna(-1).astype(int)
        df['Bene_Id_Mapped'] = df['Bene_Id'].map(global_mapping).fillna(-1).astype(int)
    return df_list, global_mapping

df_list, global_mapping = map_nodes_global([train_df, val_df, test_df])
num_nodes_global = len(global_mapping)

mapping_file = "gat_global_mapping.json"
with open(mapping_file, 'w') as f:
    json.dump(global_mapping, f)

print(f"Global mapping saved to {mapping_file}")

class TemporalEdgeDataset(Dataset):
    def __init__(self, df, seq_length):
        self.df = df
        self.seq_length = seq_length
        self.grouped = self.df.groupby(["Sender_Id_Mapped", "Bene_Id_Mapped"])

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        sender, receiver = list(self.grouped.groups.keys())[idx]
        group_data = self.grouped.get_group((sender, receiver)).sort_values("Time_step")
        
        features = group_data[["Scaled_USD_Amount", "Suspicious_Score"]].values[-self.seq_length:]
        label = group_data["Label"].iloc[-1]
        features = np.pad(features, ((self.seq_length - len(features), 0), (0, 0)), mode='constant')

        return {
            "sender": sender,
            "receiver": receiver,
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

seq_length = 10
train_dataset = TemporalEdgeDataset(train_df, seq_length)
val_dataset = TemporalEdgeDataset(val_df, seq_length)
test_dataset = TemporalEdgeDataset(test_df, seq_length)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class TemporalGraphModelGATBatch(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, gnn_hidden_dim, edge_attr_dim, num_nodes_global, heads=2):
        super().__init__()
        self.num_nodes_global = num_nodes_global
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(self.num_nodes_global, embedding_dim)

        self.gnn = GATConv(embedding_dim, gnn_hidden_dim, heads=heads, concat=True)  
        self.gnn_output_dim = gnn_hidden_dim * heads
        self.lstm = nn.LSTM(input_size=edge_attr_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.edge_predictor = nn.Linear((self.gnn_output_dim * 2) + lstm_hidden_dim, 1)

    def forward(self, sender, receiver, edge_features, edge_index):
        max_edge_index = edge_index.max().item()
        if max_edge_index >= self.node_embeddings.num_embeddings:
            self._resize_embeddings(max_edge_index + 1)

        x = self.node_embeddings.weight
        gnn_out = self.gnn(x, edge_index)  
        lstm_out, _ = self.lstm(edge_features) 
        lstm_edge_out = lstm_out[:, -1, :] 
        source_embeddings = gnn_out[sender]
        target_embeddings = gnn_out[receiver]
        combined = torch.cat([source_embeddings, target_embeddings, lstm_edge_out], dim=1)

        return self.edge_predictor(combined).squeeze(-1)

    def _resize_embeddings(self, new_num_embeddings):
        current_embeddings = self.node_embeddings.weight.data
        current_num_embeddings = current_embeddings.size(0)

        if new_num_embeddings > current_num_embeddings:
            new_embeddings = torch.nn.init.xavier_uniform_(torch.empty(new_num_embeddings, self.embedding_dim))
            new_embeddings[:current_num_embeddings] = current_embeddings
            self.node_embeddings.weight.data = new_embeddings

model = TemporalGraphModelGATBatch(
    embedding_dim=16, 
    lstm_hidden_dim=32, 
    lstm_layers=2, 
    gnn_hidden_dim=64, 
    edge_attr_dim=2, 
    num_nodes_global=num_nodes_global,
    heads=2  
).to(device)


def create_edge_index(df):
    edge_index = torch.tensor(
        np.vstack((df['Sender_Id_Mapped'].values, df['Bene_Id_Mapped'].values)),
        dtype=torch.long
    )
    return edge_index


optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
train_edge_index = create_edge_index(train_df).to(device)
val_edge_index = create_edge_index(val_df).to(device)
test_edge_index = create_edge_index(test_df).to(device)

def train_epoch(model, loader, optimizer, criterion, edge_index):
    model.train()
    total_loss = 0
    for batch in loader:
        sender = batch['sender'].to(device)
        receiver = batch['receiver'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        predictions = model(sender, receiver, features, edge_index)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)
def evaluate_model(model, loader, edge_index):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            sender = batch['sender'].to(device)
            receiver = batch['receiver'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            predictions = model(sender, receiver, features, edge_index)
            all_labels.append(labels.cpu())
            all_preds.append(torch.sigmoid(predictions).cpu())  
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_preds_binary = (all_preds > 0.5).astype(int)  
    f1 = f1_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall = recall_score(all_labels, all_preds_binary)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float('nan')  
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return f1, precision, recall, auc, fpr, fnr
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_index=train_edge_index)
    val_f1, val_precision, val_recall, val_auc, val_fpr, val_fnr = evaluate_model(model, val_loader, edge_index=val_edge_index)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation FPR: {val_fpr:.4f}, FNR: {val_fnr:.4f}")
model_save_path = "gat.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

test_f1, test_precision, test_recall, test_auc, test_fpr, test_fnr = evaluate_model(model, test_loader, edge_index=test_edge_index)
print(f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test FPR: {test_fpr:.4f}, FNR: {test_fnr:.4f}")