import os
import csv
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kafka import KafkaConsumer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from collections import deque
import json
import logging
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  
from io import BytesIO

logging.basicConfig(
    filename='Log/consumer_rule_based.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

tensorboard_logdir = "Rule_Based/tensorboard_logs/"
writer = SummaryWriter(tensorboard_logdir) 
logging.info("Starting the script.")

cross_border = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'cross_border')
country_risk = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'country_risk')
pep_involvement = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pep_involvement')
transaction_type = ctrl.Antecedent(np.arange(0, 4, 1), 'transaction_type')  
transaction_amount = ctrl.Antecedent(np.arange(0, 10001, 100), 'transaction_amount')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
sender_anonymous = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sender_anonymous')



sender_anonymous['no'] = fuzz.trapmf(sender_anonymous.universe, [0, 0, 0.3, 0.5])
sender_anonymous['yes'] = fuzz.trapmf(sender_anonymous.universe, [0.5, 0.7, 1, 1])
pep_involvement['no'] = fuzz.gbellmf(pep_involvement.universe, 0.2, 4, 0.2)
pep_involvement['yes'] = fuzz.gbellmf(pep_involvement.universe, 0.2, 4, 0.8)
transaction_type['crypto_transfer'] = fuzz.gbellmf(transaction_type.universe, 0.05, 8, 0)
transaction_type['kyc_create_account'] = fuzz.gbellmf(transaction_type.universe, 0.2, 4, 1)
transaction_type['kyc_add_account_owner'] = fuzz.gbellmf(transaction_type.universe, 0.2, 4, 2)
transaction_type['other'] = fuzz.gbellmf(transaction_type.universe, 0.2, 4, 3)
transaction_amount['low'] = fuzz.trapmf(transaction_amount.universe, [0, 0, 3800, 4000])
transaction_amount['medium'] = fuzz.trapmf(transaction_amount.universe, [3800, 4000, 6000, 7000])
transaction_amount['high'] = fuzz.trapmf(transaction_amount.universe, [6000, 8000, 10000, 10000])
country_risk['low'] = fuzz.gbellmf(country_risk.universe, 0.2, 4, 0.2)
country_risk['high'] = fuzz.gbellmf(country_risk.universe, 0.2, 4, 0.8)
cross_border['low'] = fuzz.gbellmf(cross_border.universe, 0.2, 4, 0.2)
cross_border['high'] = fuzz.gbellmf(cross_border.universe, 0.2, 4, 0.8)
risk['low'] = fuzz.gbellmf(risk.universe, 25, 2, 20)
risk['medium'] = fuzz.gbellmf(risk.universe, 20, 2, 45)
risk['high'] = fuzz.gbellmf(risk.universe, 15, 2, 75)



rules = [
    # High risk for sender being anonymous with risky transaction types or high PEP involvement or high country risk
    ctrl.Rule(sender_anonymous['yes'] & (transaction_type['crypto_transfer'] | transaction_type['other']) | (pep_involvement['yes'] | country_risk['high']), risk['high']),
    
    # Low risk for sender being anonymous but doing less risky types of transactions
    ctrl.Rule(sender_anonymous['yes'] & (transaction_type['kyc_create_account'] | transaction_type['kyc_add_account_owner']), risk['low']),
    
    # High risk for high transaction amounts combined with PEP involvement or high country risk
    ctrl.Rule(transaction_amount['high'] & (pep_involvement['yes'] | country_risk['high']), risk['high']),
    
    # Low risk for low transaction amounts, sender not anonymous, and low country risk
    ctrl.Rule(transaction_amount['low'] & sender_anonymous['no'] & country_risk['low'], risk['low']),
    
    # Low risk for cross-border transactions with low transaction amount
    ctrl.Rule(cross_border['low'] & transaction_amount['low'], risk['low']),
    
    # High risk for cross-border transactions with high transaction amount
    ctrl.Rule(cross_border['high'] & transaction_amount['high'], risk['high']),
    
    # Medium risk for medium transaction amount and medium or low PEP involvement and country risk
    ctrl.Rule(transaction_amount['medium'] & (pep_involvement['no'] & country_risk['low']), risk['medium']),
    
    # Medium risk for medium transaction amount with cross-border transaction
    ctrl.Rule(transaction_amount['medium'] & cross_border['high'], risk['medium']),
    
    # Medium risk for medium transaction amount with anonymous sender
    ctrl.Rule(transaction_amount['medium'] & sender_anonymous['yes'], risk['medium'])
]

aml_control = ctrl.ControlSystem(rules)
aml_sim = ctrl.ControlSystemSimulation(aml_control)
def create_consumer():
    try:
        consumer = KafkaConsumer(
            'rule-based',
            bootstrap_servers=[''],
            auto_offset_reset='latest', 
            enable_auto_commit=False, 
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=100000
        )
        logging.info("Kafka consumer created successfully.")
        return consumer
    except Exception as e:
        logging.error(f"Error creating Kafka consumer: {e}")
        raise

def evaluate_transaction(transaction):
    try:
        usd_amount = float(transaction.get('USD_Amount', '0'))  
        transaction_type_value = transaction.get('Transaction_Type', 'OTHER')
        sender_account = transaction.get('Sender_Account', 'Not Provided')
        bene_is_pep = transaction.get('Bene_Is_Pep', '0')  
        sender_is_pep = transaction.get('Sender_Is_Pep', '0')  
        sender_country = transaction.get('Sender_Country', 'Not Provided')
        bene_country = transaction.get('Bene_Country', 'Not Provided')

        sender_anonymous_status = 1 if sender_account == 'Not Provided' else 0

        sender_country = '0' if sender_country == 'Not Provided' else sender_country
        bene_country = '0' if bene_country == 'Not Provided' else bene_country

        pep_involvement_value = 1 if bene_is_pep == '1' or sender_is_pep == '1' else 0
        high_risk_countries = ['Iran', 'Syria', 'North-Korea', 'South Africa', 'Panama']

        cross_border_value = 1 if sender_country != bene_country and sender_country != '0' and bene_country != '0' else 0
        country_risk_value = 1 if (bene_country in high_risk_countries and bene_country != '0')  or (sender_country in high_risk_countries)else 0
        transaction_type_mapping = {
            'CRYPTO-TRANSFER': 0,
            'KYC-CREATE-ACCOUNT': 1,
            'KYC-ADD-ACCOUNT-OWNER': 2,
            'OTHER': 3
        }
        transaction_type_fuzzy_value = transaction_type_mapping.get(transaction_type_value, 3)  
        aml_sim.input['transaction_type'] = transaction_type_fuzzy_value
        aml_sim.input['pep_involvement'] = pep_involvement_value
        aml_sim.input['cross_border'] = int(cross_border_value)
        aml_sim.input['country_risk'] = country_risk_value
        aml_sim.input['sender_anonymous'] = sender_anonymous_status
        aml_sim.input['transaction_amount'] = usd_amount

        aml_sim.compute()

        risk_score = aml_sim.output['risk']

        label = 1 if risk_score >= 50 else 0  
        return label, risk_score

    except Exception as e:
        return 0, 0

def store_metrics(metrics, metrics_file):
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if f.tell() == 0:
            writer.writerow(['Batch_Number', 'F1_Score', 'Precision', 'Recall', 'Confusion_Matrix', 'AUC', 'False_Positive_Rate', 'False_Negative_Rate'])
        
        writer.writerow([
            metrics['batch_number'], 
            metrics['f1_score'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['confusion_matrix'], 
            metrics['auc'], 
            metrics['false_positive_rate'], 
            metrics['false_negative_rate']  
        ])
    
    logging.info("Metrics stored successfully.")


def store_results(results, results_file):
    try:
        logging.info(f"Storing results to {results_file}")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Results stored successfully in {results_file}.")
    except Exception as e:
        logging.error(f"Error storing results as CSV: {e}")
        raise


def process_batch(sliding_window, true_labels, risk_scores, results, all_f1_scores, batch_number, metrics_file, output_folder):
    f1 = f1_score(true_labels, sliding_window)
    precision = precision_score(true_labels, sliding_window)
    recall = recall_score(true_labels, sliding_window)
    
    conf_matrix = confusion_matrix(true_labels, sliding_window)
    tn, fp, fn, tp = conf_matrix.ravel()  
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    fpr, tpr, _ = roc_curve(true_labels, risk_scores)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "batch_number": batch_number,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix.tolist(),  
        "auc": roc_auc,
        "false_positive_rate": fpr_value, 
        "false_negative_rate": fnr_value  
    }

    store_metrics(metrics, metrics_file)
    results_file = os.path.join(output_folder, f'results_batch_{batch_number}.csv')
    store_results(results, results_file)
    logging.info(f"Results stored for batch number {batch_number}.")

    writer.add_scalar("F1 Score", f1, batch_number)
    writer.add_scalar("Precision", precision, batch_number)
    writer.add_scalar("Recall", recall, batch_number)
    writer.add_scalar("AUC ROC", roc_auc, batch_number)
    writer.add_scalar("False Positive Rate", fpr_value, batch_number)
    writer.add_scalar("False Negative Rate", fnr_value, batch_number)

    writer.flush()

    
def consume_and_evaluate(window_size=5000, output_folder="Rule_Based"):
    consumer = create_consumer()
    sliding_window = deque(maxlen=window_size)
    true_labels = deque(maxlen=window_size)
    risk_scores = deque(maxlen=window_size)
    results = []
    count = 0
    batch_number = 0
    metrics_file = os.path.join(output_folder, 'metrics.csv')

    try:
        for message in consumer:
            transactions = message.value
            if not transactions:
                logging.warning("Received an empty or non-list message. Skipping.")
                continue

            for transaction in transactions:  
                if isinstance(transaction, dict):
                    label, risk_score = evaluate_transaction(transaction)
                    true_label = transaction.get('Label', 0)  
                    sliding_window.append(label)
                    true_labels.append(true_label)
                    risk_scores.append(risk_score)
                    results.append({
                        "Transaction_ID": transaction.get('Transaction_Id', 'Unknown'),
                        "Transaction_Type": transaction.get('Transaction_Type', 'Unknown'),
                        "Predicted_Label": label,
                        "Risk_Score": risk_score,
                        "True_Label": true_label
                    })
                    count += 1

                else:
                    logging.error("Received transaction is not a dictionary.")

            logging.info(f"Processed {len(transactions)} transactions, total processed: {count}")

            if count % window_size == 0:
                process_batch(sliding_window, true_labels, risk_scores, results, [], batch_number, metrics_file, output_folder)
                results = []
                batch_number += 1

        if results:
            process_batch(sliding_window, true_labels, risk_scores, results, [], batch_number, metrics_file, output_folder)
            logging.info("Final batch processed.")

    except Exception as e:
        logging.error(f"Error during consumption and evaluation: {e}")

    finally:
        consumer.close()  
    
if __name__ == "__main__":
    try:
        consume_and_evaluate()
    except Exception as e:
        logging.critical(f"Critical error occurred: {e}")