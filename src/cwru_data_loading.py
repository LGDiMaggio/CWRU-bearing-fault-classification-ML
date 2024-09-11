import os
import kaggle
import scipy.io
import pandas as pd
import re
from feature_extraction import extract_time_domain_features, extract_frequency_domain_features

def download_dataset(dataset, data_path='../data'):
    """Download dataset from Kaggle to specified directory."""
    # Ensure that the KAGGLE_CONFIG_DIR environment variable is set outside this script
    
    # Check if the specified data directory exists, if not, create it
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Download the dataset
    kaggle.api.dataset_download_files(dataset, path=data_path, unzip=True)
    
    print(f"Dataset downloaded and extracted to: {data_path}")

def load_signals(file_path):
    """Carica tutti i segnali DE_time e FE_time da un file MAT."""
    data = scipy.io.loadmat(file_path)
    DE_signals = [data[k].squeeze() for k in data if 'DE_time' in k]
    FE_signals = [data[k].squeeze() for k in data if 'FE_time' in k]
    return DE_signals, FE_signals

def segment_signal(signal, frame_size=2048, overlap=0.5):
    """Divide il segnale in segmenti con una data lunghezza e overlap."""
    step_size = int(frame_size * (1 - overlap))
    segments = [signal[i:i + frame_size] for i in range(0, len(signal) - frame_size + 1, step_size)]
    return segments

def parse_filename(file_name):
    """Estrae le informazioni dai nomi dei file distinguendo correttamente tipo e dimensione del danno."""
    if "Normal" in file_name:
        health_state = "Normal"
        damage_size = "0"
    else:
        match = re.match(r"([a-zA-Z]+)(\d+)", file_name)
        health_state, damage_size = match.groups() if match else ("Unknown", "Unknown")
    return health_state, damage_size

def build_dataset2(directory, frame_size=2048, overlap=0.5, feature_type='both', sampling_rate=48000):
    dataset = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            DE_signals, FE_signals = load_signals(file_path)
            health_state, damage_size = parse_filename(file_name)
            for signal_type, signals in zip(['DE', 'FE'], [DE_signals, FE_signals]):
                for signal in signals:
                    segments = segment_signal(signal, frame_size, overlap)
                    for segment in segments:
                        if feature_type == 'time' or feature_type == 'both':
                            features = extract_time_domain_features(segment)
                        if feature_type == 'frequency' or feature_type == 'both':
                            freq_features = extract_frequency_domain_features(segment, sampling_rate)
                            features.update(freq_features) if feature_type == 'both' else features.update(freq_features)

                        specific_label = f"{health_state}{damage_size}{signal_type}"
                        data_row = [specific_label, health_state, damage_size, signal_type, *features.values()]
                        dataset.append(data_row)
    
    columns = ['Specific Label', 'Health State', 'Damage Size', 'Signal Type'] + list(features.keys())
    return pd.DataFrame(dataset, columns=columns)

def build_dataset(directory, frame_size=2048, overlap=0.5, feature_type='both', sampling_rate=48000):
    dataset = []
    features_list = []  # Initialize features_list here to store feature names
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            DE_signals, FE_signals = load_signals(file_path)
            health_state, damage_size = parse_filename(file_name)
            for signal_type, signals in zip(['DE', 'FE'], [DE_signals, FE_signals]):
                for signal in signals:
                    segments = segment_signal(signal, frame_size, overlap)
                    for segment in segments:
                        features = {}
                        if feature_type == 'time' or feature_type == 'both':
                            features.update(extract_time_domain_features(segment))
                        if feature_type == 'frequency' or feature_type == 'both':
                            freq_features = extract_frequency_domain_features(segment, sampling_rate)
                            features.update(freq_features)
                        
                        # Ensure features_list is updated correctly
                        if not features_list:
                            features_list = list(features.keys())

                        specific_label = f"{health_state}{damage_size}{signal_type}"
                        data_row = [specific_label, health_state, damage_size, signal_type] + list(features.values())
                        dataset.append(data_row)
    
    columns = ['Specific Label', 'Health State', 'Damage Size', 'Signal Type'] + features_list
    return pd.DataFrame(dataset, columns=columns), features_list
