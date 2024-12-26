from kafka import KafkaProducer
import json
import pandas as pd
import time

producer = KafkaProducer(
    bootstrap_servers=[''],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    acks='all', 
    retries=10,  
    retry_backoff_ms=100  
)

def send_data_with_count_window(filename, batch_size=100, window_size=100000, delay_seconds=2):
    df = pd.read_csv(filename)
    df = df.sort_values(by='Time_step')
    window = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    current_window_size = 0
    window_number = 1
    
    for index, row in df.iterrows():
        window.append(row.to_dict())
        current_window_size += 1
        
        if current_window_size >= window_size:
            try:
                producer.send('hybrid-gcn-lstm1', value=window)
                producer.flush()  
                print(f"Successfully sent window {window_number} with {current_window_size} records")
            except Exception as e:
                print(f"Error sending window {window_number}: {str(e)}")
            
            window.clear()
            current_window_size = 0
            window_number += 1
            
            time.sleep(delay_seconds)
    
    if window:
        try:
            producer.send('hybrid-gcn-lstm1', value=window)
            producer.flush() 
            print(f"Successfully sent final window with {current_window_size} records")
        except Exception as e:
            print(f"Error sending final window: {str(e)}")

send_data_with_count_window('Thesis/processed_test_gcn.csv', batch_size=100, window_size=500)
