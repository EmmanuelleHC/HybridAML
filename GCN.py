import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_paths = {
    "train": {"raw": "Thesis/train.csv", "processed": "Thesis/procesed_train.csv"},
    "val": {"raw": "Thesis/val.csv", "processed": "Thesis/processed_val.csv"},
    "test": {"raw": "Thesis/test.csv", "processed": "Thesis/processed_test.csv"},
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
    if is_train:
        df["Scaled_USD_Amount"] = scaler.fit_transform(df[["USD_amount"]])
    else:
        df["Scaled_USD_Amount"] = scaler.transform(df[["USD_amount"]])

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
                raw_path, processed_path, scaler=scaler, downsample=downsample, is_train=is_train
            )
        else:
            df = preprocess_and_load_data(
                raw_path, processed_path, scaler=scaler, downsample=downsample, is_train=is_train
            )
        
        datasets[split] = {"df": df}

    return datasets

datasets = process_datasets(data_paths, scaler)
train_df = datasets["train"]["df"]
val_df = datasets["val"]["df"]
test_df = datasets["test"]["df"]

def map_nodes_global(df_list):
    global_mapping = {}
    for df in df_list:
        for col in ["Sender_Id", "Bene_Id"]:
            for node in df[col]:
                if node not in global_mapping:
                    global_mapping[node] = len(global_mapping)
    for df in df_list:
        df["Sender_Id_Mapped"] = df["Sender_Id"].map(global_mapping)
        df["Bene_Id_Mapped"] = df["Bene_Id"].map(global_mapping)
    return df_list, global_mapping

df_list, global_mapping = map_nodes_global([train_df, val_df, test_df])
num_nodes_global = len(global_mapping)

def create_graph_dataset(df, seq_length):
    graphs = []
    grouped = df.groupby(["Sender_Id_Mapped", "Bene_Id_Mapped"])
    
    for (sender, receiver), group in grouped:
        group_data = group.sort_values("Time_step")
        features = group_data[["Scaled_USD_Amount"]].values[-seq_length:]
        label = group_data["Label"].iloc[-1]
        features = np.pad(features, ((seq_length - len(features), 0), (0, 0)), mode='constant')

        if sender >= num_nodes_global or receiver >= num_nodes_global:
            print(f"Warning: Node indices out of bounds! Sender: {sender}, Receiver: {receiver}, Max index: {num_nodes_global-1}")
            continue

        edge_index = torch.tensor([[sender, receiver]], dtype=torch.long).t()
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float).unsqueeze(0)

        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


seq_length = 10
train_graphs = create_graph_dataset(train_df, seq_length)
val_graphs = create_graph_dataset(val_df, seq_length)
test_graphs = create_graph_dataset(test_df, seq_length)

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
class TemporalGraphModelGCNBatch(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, gcn_hidden_dim, edge_attr_dim, num_nodes_global):
        super().__init__()
        self.num_nodes_global = num_nodes_global
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(self.num_nodes_global, embedding_dim)
        self.lstm = nn.LSTM(input_size=edge_attr_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.gcn = GCNConv(embedding_dim, gcn_hidden_dim)  # GCN layer
        
        self.edge_predictor = nn.Linear((gcn_hidden_dim * 2) + lstm_hidden_dim, 1)

    def forward(self, data):
        max_edge_index = data.edge_index.max().item()
        if max_edge_index >= self.node_embeddings.num_embeddings:
            self._resize_embeddings(max_edge_index + 1)

        x = self.node_embeddings.weight
        gcn_out = self.gcn(x, data.edge_index) 
        lstm_input = data.x.view(data.x.size(0), -1, 1) 
        lstm_out, _ = self.lstm(lstm_input)
        lstm_edge_out = lstm_out[:, -1, :][:data.edge_index.size(1), :]  
        source_embeddings = gcn_out[data.edge_index[0]]
        target_embeddings = gcn_out[data.edge_index[1]]
        combined = torch.cat([source_embeddings, target_embeddings, lstm_edge_out], dim=1)
        return self.edge_predictor(combined).squeeze(-1)

    def _resize_embeddings(self, new_num_embeddings):
        current_embeddings = self.node_embeddings.weight.data
        current_num_embeddings = current_embeddings.size(0)
        
        if new_num_embeddings > current_num_embeddings:
            new_embeddings = torch.nn.init.xavier_uniform_(torch.empty(new_num_embeddings, self.embedding_dim))
            new_embeddings[:current_num_embeddings] = current_embeddings
            self.node_embeddings.weight.data = new_embeddings

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


num_nodes_global = len(global_mapping)  
model = TemporalGraphModelGCNBatch(
    embedding_dim=16, 
    lstm_hidden_dim=32, 
    lstm_layers=2, 
    gcn_hidden_dim=64, 
    edge_attr_dim=1, 
    num_nodes_global=num_nodes_global
).to(device)


optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


def evaluate_model_with_cm(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []  
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device) 
            outputs = model(batch)
            targets = batch.y.float()  
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()  
            preds = (probs > 0.5).astype(int)  
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float('nan') 
    cm = confusion_matrix(all_targets, all_preds)

    return avg_loss, f1, precision, recall, auc, cm

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_f1, val_precision, val_recall, val_auc, val_cm = evaluate_model_with_cm(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

test_loss, test_f1, test_precision, test_recall, test_auc, test_cm = evaluate_model_with_cm(model, test_loader, criterion)

tn, fp, fn, tp = test_cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Test Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, AUC: {test_auc:.4f}")
print(f"Test FPR: {fpr:.4f}, Test FNR: {fnr:.4f}")