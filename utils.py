import json
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

mapping = {'1AVB': '1AVB',
           'AF': 'AFLT', 
           'AFIB': 'AFIB', 
           'APB': 'PAC', 
           'AQW': 'ABQRS', 
           'IDC': 'IVCD',
            'LVH': 'LVH', 
            'LVQRSAL': 'LVOLT', 
            'RBBB': 'CRBBB', 
            'SR': 'SR', 
            'ST': 'STACH',
              'STDD': 'STD_', 
              'STE': 'STE_', 
              'STTC': 'NST_', 
              'SVT': 'SVTAC', 
              'TWC': 'NT_',
              'TWO': 'INVT'}
           

def load_X_y(outcome, lead_name=None):
    data_df = pd.read_pickle('data/arrythmia_dataset.pickle')
    X, y = data_df['wf'].to_numpy(), data_df[[outcome]].to_numpy()
    y = y.astype(float)
    X = np.stack(X, axis=0)

    if lead_name is not None:
        lead_labels = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_idx = lead_labels.index(lead_name)
        X = X[:, :, lead_idx].reshape(-1, 5000, 1)

    return X, y

def split_train_val_test(X, y, train_size, val_size, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size/(1-train_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def get_arrythmia_path():
    with open("config.json", 'r') as f:
        data = json.load(f)
    
    return data["arrythmia_path"]