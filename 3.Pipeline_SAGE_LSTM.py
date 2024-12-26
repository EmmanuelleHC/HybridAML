import os
import json
import logging
import pandas as pd
import numpy as np
from collections import deque
import csv
import torch
import torch.nn as nn
from kafka import KafkaConsumer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
import torch_geometric.nn as pyg_nn
from itertools import product
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import ParameterGrid
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import anfis
from anfis.membership import BellMembFunc, GaussMembFunc
from anfis.membership import make_bell_mfs, make_gauss_mfs 
from anfis.anfis import AnfisNet
from torch_geometric.data import Data
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_folder="Hybrid_SAGE_LSTM"

writer = SummaryWriter(log_dir='Hybrid_SAGE_LSTM/tensorboard_logs/')

logging.basicConfig(
    filename='Log/consumer_hybrid_based.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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


def load_anfis_model(file_path, memFuncs, scaled_data, target_data):
    Y = torch.tensor(target_data, dtype=torch.float32)

    X = torch.tensor(
        scaled_data[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values,
        dtype=torch.float32
    ).to(device)

    anfis = ANFIS(X, Y, memFuncs).to(device)

    anfis.load_state_dict(torch.load(file_path))
    anfis.eval()
    
    logging.info(f"ANFIS model loaded from {file_path}")
    return anfis


def add_suspicious_score(df, anfis_model):
    scores = anfis_model.forward(torch.tensor(df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32).to(device)).detach().cpu().numpy()
    df["Suspicious_Score"] = scores
    return df

mapping_file = "sage_global_mapping.json"

if os.path.exists(mapping_file):
    with open(mapping_file, 'r') as f:
        global_mapping = json.load(f)
else:
    global_mapping = {}

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
def create_edge_index(df):

    edge_index = torch.tensor(
        np.vstack((df['Sender_Id_Mapped'].values, df['Bene_Id_Mapped'].values)),
        dtype=torch.long
    )
    return edge_index
class TemporalGraphModelSAGEBatch(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_layers, gcn_hidden_dim, edge_attr_dim, num_nodes_global):
        super().__init__()
        self.num_nodes_global = num_nodes_global
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(self.num_nodes_global, embedding_dim)
        self.lstm = nn.LSTM(input_size=edge_attr_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.gcn = SAGEConv(embedding_dim, gcn_hidden_dim) 
        self.edge_predictor = nn.Linear((gcn_hidden_dim * 2) + lstm_hidden_dim, 1)

    def forward(self, sender, receiver, edge_features, edge_index):
        max_edge_index = edge_index.max().item()
        if max_edge_index >= self.node_embeddings.num_embeddings:
            self._resize_embeddings(max_edge_index + 1)

        x = self.node_embeddings.weight
        gcn_out = self.gcn(x, edge_index)  
        lstm_out, _ = self.lstm(edge_features) 
        lstm_edge_out = lstm_out[:, -1, :]  
        source_embeddings = gcn_out[sender]
        target_embeddings = gcn_out[receiver]
        combined = torch.cat([source_embeddings, target_embeddings, lstm_edge_out], dim=1)
        
        return self.edge_predictor(combined).squeeze(-1)

    def _resize_embeddings(self, new_num_embeddings):
        current_embeddings = self.node_embeddings.weight.data
        current_num_embeddings = current_embeddings.size(0)
        
        if new_num_embeddings > current_num_embeddings:
            new_embeddings = torch.nn.init.xavier_uniform_(torch.empty(new_num_embeddings, self.embedding_dim))
            new_embeddings[:current_num_embeddings] = current_embeddings
            self.node_embeddings.weight.data = new_embeddings

def create_consumer():
    try:
        consumer = KafkaConsumer(
            'hybrid-sage-lstm',
            bootstrap_servers=[''],
            auto_offset_reset='latest', 
            enable_auto_commit=False,  
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000000
        )
        logging.info("Kafka consumer created successfully.")
        return consumer
    except Exception as e:
        logging.error(f"Error creating Kafka consumer: {e}")
        raise

def store_metrics(metrics, metrics_file, batch_number):
    try:
        logging.info(f"Storing metrics for batch number {batch_number}.")
        
        writer.add_scalar('F1_Score', metrics['f1_score'], batch_number)
        writer.add_scalar('Precision', metrics['precision'], batch_number)
        writer.add_scalar('Recall', metrics['recall'], batch_number)
        writer.add_scalar('AUC', metrics['auc'], batch_number)
        writer.add_scalar("False Positive Rate", metrics['false_positive_rate'], batch_number)
        writer.add_scalar("False Negative Rate", metrics['false_negative_rate'], batch_number)
        cm = metrics['confusion_matrix']
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for Batch {batch_number}')
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        writer.add_figure(f'Confusion_Matrix/Batch_{batch_number}', figure)
        plt.close(figure)

        logging.info(f"Metrics stored successfully for batch number {batch_number}.")

        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer_2 = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer_2.writeheader() 
            writer_2.writerow(metrics)
        logging.info(f"Metrics successfully written to {metrics_file}")

    except Exception as e:
        logging.error(f"Error storing metrics: {e}")
        raise

def store_results(results, results_file):
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Results stored successfully in {results_file}.")
    except Exception as e:
        logging.error(f"Error storing results as CSV: {e}")
        raise
def process_batch(sliding_window, true_labels, results, all_f1_scores, batch_number, metrics_file, output_folder, writer):
    try:
        logging.info(f"Processing batch number {batch_number}.")

        threshold = 0.5
        binary_predictions = [1 if pred >= threshold else 0 for pred in sliding_window]

        f1 = f1_score(true_labels, binary_predictions)
        precision = precision_score(true_labels, binary_predictions)
        recall = recall_score(true_labels, binary_predictions)
        conf_matrix = confusion_matrix(true_labels, binary_predictions).tolist()
        all_f1_scores.append(f1) 
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()

        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0

      
        fpr, tpr, _ = roc_curve(true_labels, sliding_window) 
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, sliding_window)  
        pr_auc = auc(recall_curve, precision_curve)

        metrics = {
            "batch_number": batch_number,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "auc": roc_auc,  
            "pr_auc": pr_auc,  
            "false_positive_rate": fpr_value, 
            "false_negative_rate": fnr_value   
        }

        store_metrics(metrics, metrics_file, batch_number)
        logging.info(f"Metrics stored for batch number {batch_number}.")

        results_file = os.path.join(output_folder, f'results_batch_{batch_number}.csv')
        store_results(results, results_file)
        logging.info(f"Results stored for batch number {batch_number}.")

    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        raise

from torch_geometric.data import Batch
def consume_and_evaluate(global_mapping, model_path, window_size=5000, output_folder="Hybrid_SAGE_LSTM", timeout=3800, batch_size=500):
    writer = SummaryWriter(output_folder)

    sliding_window = deque(maxlen=window_size)
    true_labels = deque(maxlen=window_size)
    all_f1_scores = []
    results = []
    batch_data = []
    count = 0
    batch_number = 0
    last_message_time = time.time()

    logging.info("Loading models and components...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes_global = len(global_mapping)
   
    model = TemporalGraphModelSAGEBatch(
    embedding_dim=16, 
    lstm_hidden_dim=32, 
    lstm_layers=2, 
    gcn_hidden_dim=64, 
    edge_attr_dim=2,  
    num_nodes_global=num_nodes_global
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("SAGE model loaded successfully.")

    anfis_model_path = "sage_anfis_model.pth"
    scaler = joblib.load('hybrid_sage_scaler.pkl')

    def update_and_apply_mapping(df, global_mapping, mapping_file):
        max_id = max(global_mapping.values(), default=0)

        for id_col in ["Sender_Id", "Bene_Id"]:
            for id_value in df[id_col].unique():
                if id_value not in global_mapping:
                    max_id += 1
                    global_mapping[id_value] = max_id

        with open(mapping_file, 'w') as f:
            json.dump(global_mapping, f)

        df["Sender_Id_Mapped"] = df["Sender_Id"].map(global_mapping).fillna(-1).astype(int)
        df["Bene_Id_Mapped"] = df["Bene_Id"].map(global_mapping).fillna(-1).astype(int)
        return df

    logging.info("Creating Kafka consumer...")
    consumer = create_consumer()

    try:
        for message in consumer:
            transactions = message.value 
            logging.info(f"Transactions received in this message: {len(transactions)}")
            if transactions:
                batch_data.extend(transactions)
                logging.info(f"Batch data size after appending: {len(batch_data)}")

                last_message_time = time.time()

                if len(batch_data) >= batch_size:
                    batch_df = pd.DataFrame(batch_data[:batch_size])  
                    batch_data = batch_data[batch_size:]  
                    batch_df = update_and_apply_mapping(batch_df, global_mapping, mapping_file)
                    logging.info("Mapping applied to batch_df.")

                    scaled_data = scaler.transform(batch_df[["USD_amount"]])
                    batch_df["Scaled_USD_Amount"] = scaled_data

                    memFuncs = MemFuncs(torch.tensor(batch_df[["Country_Risk", "Cross_Border", "PEP_Involvement"]].values, dtype=torch.float32).to(device))
                    logging.info("MemFuncs initialized for the batch.")
                    target_data = batch_df["Label"].values  
                    anfis_model = load_anfis_model(anfis_model_path, memFuncs, batch_df, target_data)

                    logging.info("ANFIS model loaded successfully with dynamic MemFuncs.")

                    df = add_suspicious_score(batch_df, anfis_model)

                    edge_index = create_edge_index(df)
                    logging.info(f"Edge index created with shape: {edge_index.shape}")

                    dataset = TemporalEdgeDataset(df, seq_length=10)
                    loader = DataLoader(dataset, batch_size=64, shuffle=False)
                    with torch.no_grad():
                        for batch in loader:
                            logging.info("Processing batch...")

                            sender = batch['sender'].to(device)
                            receiver = batch['receiver'].to(device)
                            features = batch['features'].to(device)
                            labels = batch['label'].to(device)

                            predictions = model(sender, receiver, features, edge_index.to(device))
                            
                            sliding_window.extend(predictions.cpu().numpy())
                            true_labels.extend(labels.cpu().numpy())

                            results.extend([
                                {
                                    "sender": sender[i].item(),
                                    "receiver": receiver[i].item(),
                                    "true_label": labels[i].item(),
                                    "predicted_label": predictions[i].item()
                                } for i in range(len(labels))
                            ])
                        count += len(batch_df)

                        if count >= window_size or (time.time() - last_message_time > timeout):
                            process_batch(
                                list(sliding_window),
                                list(true_labels),
                                results,
                                all_f1_scores,
                                batch_number,
                                os.path.join(output_folder, "metrics.csv"),
                                output_folder,
                                writer
                            )

                            results = []
                            batch_number += 1
                            count = 0

    except Exception as e:
        logging.error(f"Error during consumption and evaluation: {e}")
        raise
    finally:
        writer.close()

if __name__ == "__main__":
    try:
        model_path = 'sage.pth'
        consume_and_evaluate(global_mapping,model_path=model_path)
    except Exception as e:
        logging.critical(f"Critical error in main: {e}", exc_info=True)
