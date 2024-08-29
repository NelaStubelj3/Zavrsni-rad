import os
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from segmentation_models_pytorch.losses import DiceLoss


class MultiTaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiTaskRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN shared layer
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # Task 1: Track
        self.track_output_layer = nn.Linear(hidden_size, 1)  # cursor_in_target
        self.track_activation = nn.Sigmoid()  # Apply sigmoid for binary output

        # Task 2: Resman
        self.resman_output_layer = nn.Linear(hidden_size, 2)  # a_in_tolerance, b_in_tolerance
        self.resman_activation = nn.Sigmoid()  # Apply sigmoid for binary output

        # Task 3: Sysmon
        self.sysmon_hidden_layer = nn.Linear(hidden_size, hidden_size // 2)  # Dodatni sloj
        self.sysmon_activation = nn.ReLU()  # Promjena aktivacijske funkcije
        self.sysmon_output_layer = nn.Linear(hidden_size // 2, 2)  

    def forward(self, inputs):
        rnn_output, _ = self.rnn(inputs)
        last_output = rnn_output[:, -1, :]  
        
        # Task 1: Track
        track_output = self.track_output_layer(last_output)
        track_output = self.track_activation(track_output)  

        # Task 2: Resman
        resman_output = self.resman_output_layer(last_output)
        resman_output = self.resman_activation(resman_output)  

        # Task 3: Sysmon
        sysmon_hidden_output = self.sysmon_hidden_layer(last_output)
        sysmon_hidden_output = self.sysmon_activation(sysmon_hidden_output)
        sysmon_output = self.sysmon_output_layer(sysmon_hidden_output)        

        return track_output, resman_output, sysmon_output

class CustomDataset(Dataset):
    def __init__(self, df, input_cols, output_cols, scenario_col, sequence_length=10):
        self.df = df.reset_index(drop=True)  # Resetiranje indeksa
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.scenario_col = scenario_col
        self.sequence_length = sequence_length

        # Convert inputs to numeric values and handle non-numeric values
        self.df[self.input_cols] = self.df[self.input_cols].apply(pd.to_numeric, errors='coerce')
        self.df[self.input_cols] = self.df[self.input_cols].fillna(0)  # or you can use another strategy to handle NaNs

    def __len__(self):
        num_samples = len(self.df) - self.sequence_length + 1
        num_batches = math.ceil(num_samples / batch_size)
        return num_batches * batch_size 

    def __getitem__(self, idx):
        # Ensure the idx is within bounds
        if idx + self.sequence_length > len(self.df):
            idx = len(self.df) - self.sequence_length

        inputs = self.df.loc[idx:idx+self.sequence_length-1, self.input_cols].values
        inputs = torch.tensor(inputs.astype(float), dtype=torch.float32)

        sample = self.df.iloc[idx + self.sequence_length - 1]

        cursor_in_target = sample['cursor_in_target']
        a_in_tolerance = sample['a_in_tolerance']
        b_in_tolerance = sample['b_in_tolerance']
        signal_detection_HIT = sample['signal_detection_HIT']
        signal_detection_MISS = sample['signal_detection_MISS']

        outputs = torch.tensor([cursor_in_target, a_in_tolerance, b_in_tolerance, signal_detection_HIT, signal_detection_MISS], dtype=torch.float32)
        
        scenario = sample[self.scenario_col]

        return inputs, outputs, scenario
    
def compute_class_weights(labels):
    class_sample_counts = np.bincount(labels, minlength=2)  # Ensure at least 2 counts
    total_samples = len(labels)
    class_weights = np.zeros_like(class_sample_counts, dtype=np.float32)
    
    for i, count in enumerate(class_sample_counts):
        if count > 0:
            class_weights[i] = total_samples / (len(class_sample_counts) * count)
        else:
            class_weights[i] = 0.0  # Handle division by zero for classes with no samples
    
    return torch.tensor(class_weights, dtype=torch.float)
def compute_class_weights_for_sysmon(targets):
    #print(targets)
    class_weights = compute_class_weights(targets.flatten())
    return class_weights

def evaluate_classification_model(model, dataloader):
    model.eval()
    all_track_predictions = []
    all_track_targets = []
    all_resman_predictions = []
    all_resman_targets = []
    all_sysmon_predictions = []
    all_sysmon_targets = []

    with torch.no_grad():
        for inputs, targets, scenario in dataloader:
            track_output, resman_output, sysmon_output = model(inputs)

            # Task 1: Track
            track_predictions = torch.round(track_output).cpu().numpy().tolist()
            track_targets = targets[:, 0].cpu().numpy().tolist()
            all_track_predictions.extend(track_predictions)
            all_track_targets.extend(track_targets)

            # Task 2: Resman
            resman_predictions = (resman_output > 0.5).cpu().numpy().tolist()
            resman_targets = targets[:, 1:3].cpu().numpy().tolist()
            all_resman_predictions.extend(resman_predictions)
            all_resman_targets.extend(resman_targets)

            # Task 3: Sysmon
            sysmon_predictions = torch.argmax(sysmon_output, dim=1).cpu().numpy()
            sysmon_targets = torch.argmax(targets[:, 3:], dim=1).cpu().numpy()

            # Filter out '0' class
            valid_indices = sysmon_targets != 0
            sysmon_predictions = sysmon_predictions[valid_indices]
            sysmon_targets = sysmon_targets[valid_indices]

            all_sysmon_predictions.extend(sysmon_predictions.tolist())
            all_sysmon_targets.extend(sysmon_targets.tolist())

    return (np.array(all_track_predictions), np.array(all_track_targets),
            np.array(all_resman_predictions), np.array(all_resman_targets),
            np.array(all_sysmon_predictions), np.array(all_sysmon_targets))

def find_sequence_csv_files(root_dir):
    csv_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "sequence" in file and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                csv_files.append(file_path)
    return csv_files


def load_and_combine_csv_files(csv_files):
    combined_df = pd.DataFrame()
    for file in csv_files:
        scenario_df = pd.read_csv(file)
        if 'easy' in file:
            scenario_df['scenario'] = 'easy'
        elif 'medium' in file:
            scenario_df['scenario'] = 'medium'
        elif 'hard' in file:
            scenario_df['scenario'] = 'hard'
        else:
            scenario_df['scenario'] = 'unknown'
        
        combined_df = pd.concat([combined_df, scenario_df], ignore_index=True)
    
    return combined_df

if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 10  
    hidden_size = 64
    input_size = 19  # Total input size for combined tasks
    batch_size = 32
    sequence_length = 10
    # Root directory
    root_directory = 'OpenMATB/sessions/2024-08-15' 
    print(f"Trenutni radni direktorij: {os.getcwd()}")
    csv_files = find_sequence_csv_files(root_directory)
    print(csv_files)  # This should list all CSV files found

    # Učitaj i kombiniraj sve pronađene CSV datoteke u jedan dataframe
    combined_df = load_and_combine_csv_files(csv_files)

    # Podjela podataka na treniranje i testiranje
    train_df, test_df = train_test_split(combined_df, test_size=0.7, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Definiranje stupaca
    combined_input_cols = [
        'center_deviation', 'joystick_x', 'joystick_y',  # Track input cols
        'a_deviation', 'b_deviation', 'pump_1_flow', 'pump_2_flow', 'pump_3_flow', 'pump_4_flow', 'pump_5_flow', 'pump_6_flow', 'pump_7_flow', 'pump_8_flow',  # Resman input cols
        'scale1', 'scale2', 'scale3', 'scale4', 'event_occured', 'reacted'  # Sysmon input cols
    ]

    combined_output_cols = [
        'cursor_in_target',  # Track output cols
        'a_in_tolerance', 'b_in_tolerance',  # Resman output cols
        'signal_detection_HIT', 'signal_detection_MISS'  # Sysmon output cols
    ]

    scenario_col = 'scenario'  # Column containing scenario labels

    # Kreiranje skupa podataka za treniranje i testiranje
    train_dataset = CustomDataset(train_df, combined_input_cols, combined_output_cols, scenario_col, sequence_length=sequence_length)
    test_dataset = CustomDataset(test_df, combined_input_cols, combined_output_cols, scenario_col, sequence_length=sequence_length)

    # Kreiranje DataLoader-a
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Definiranje težina za scenarije
    scenario_weights = {'easy': 1.0, 'medium': 1.2, 'hard': 1.5}

    # Model
    model = MultiTaskRNN(input_size, hidden_size)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 


    # Option 2: Class-Balanced Loss (requires installation: pip install class-balanced-loss)
    # from class_balanced_loss import ClassBalancedLoss
    # sysmon_criterion = ClassBalancedLoss(samples_per_cls=[hit_count, miss_count])  # Provide counts for each class

    # Option 3: Dice Loss (requires installation: pip install segmentation-models-pytorch)
    #sysmon_criterion = DiceLoss(mode='binary')  # Use 'binary' for binary classification

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Additional variables to count HIT and MISS occurrences
    hit_count = 0
    miss_count = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets, scenario) in enumerate(train_loader):
            optimizer.zero_grad()
            track_output, resman_output, sysmon_output = model(inputs)

            # Separate targets for different tasks
            track_targets = targets[:, :1]
            resman_targets = targets[:, 1:3]
            
            sysmon_targets = targets[:, 3:]
            #sysmon_labels = torch.argmax(sysmon_targets, dim=1)

            sysmon_targets_onehot = torch.nn.functional.one_hot(torch.argmax(sysmon_targets, dim=1), num_classes=2).float()

            track_criterion = nn.BCEWithLogitsLoss()  

            resman_criterion = nn.BCEWithLogitsLoss()
            print("targets:", targets)

            print("sysmon_targets:", sysmon_targets)
            print("Minimalna vrijednost u sysmon_targets:", sysmon_targets.min())

            sysmon_class_weights = compute_class_weights_for_sysmon(sysmon_targets)

            sysmon_targets_indices = torch.argmax(sysmon_targets, dim=1)
            sysmon_criterion = nn.CrossEntropyLoss(weight=sysmon_class_weights)
            sysmon_loss = sysmon_criterion(sysmon_output, sysmon_targets_indices)
 

            track_loss = track_criterion(track_output.squeeze(), track_targets.squeeze())
            resman_loss = resman_criterion(resman_output, resman_targets)
            sysmon_loss = sysmon_criterion(sysmon_output, sysmon_targets_indices)  
        
            total_loss = track_loss + resman_loss + sysmon_loss
            """
             # Apply scenario weight to the loss
            scenario_weight = torch.tensor([scenario_weights[s] for s in scenario], dtype=torch.float32)
            track_loss = track_loss.unsqueeze(dim=0) * scenario_weight
            resman_loss = resman_loss.unsqueeze(dim=0) * scenario_weight
            sysmon_loss = sysmon_loss.unsqueeze(dim=0) * scenario_weight  # Reshape sysmon_loss to [1]

            # Combine losses
            total_loss = track_loss.mean() + resman_loss.mean() + sysmon_loss.mean()
            """
            
            total_loss.backward()
            optimizer.step()

            # Count HIT and MISS occurrences in the current batch
            if sysmon_targets_onehot.ndim > 1:
                hit_count += (sysmon_targets_onehot[:, 0] == 1).sum().item()
                miss_count += (sysmon_targets_onehot[:, 1] == 1).sum().item()
            else:
            # If it's a 1D tensor, compare directly
                hit_count += (sysmon_targets_onehot == 1).sum().item()
                miss_count += (sysmon_targets_onehot == 0).sum().item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

    # After training is complete, print the HIT and MISS counts
    print(f"Total HIT count: {hit_count}")
    print(f"Total MISS count: {miss_count}")

    # Evaluation phase (as provided in your initial code)
    (track_predictions, track_targets,
    resman_predictions, resman_targets,
    sysmon_predictions, sysmon_targets) = evaluate_classification_model(model, test_loader)

    print("Track Task - Accuracy:", accuracy_score(track_targets, track_predictions))
    print("Track Task - Classification Report:")
    print(classification_report(track_targets, track_predictions))

    print("Resman Task - Accuracy:", accuracy_score(resman_targets, resman_predictions))
    print("Resman Task - Classification Report:")
    print(classification_report(resman_targets, resman_predictions))

    print("Sysmon Task - Accuracy:", accuracy_score(sysmon_targets, sysmon_predictions))
    print("Sysmon Task - Classification Report:")
    print(classification_report(sysmon_targets, sysmon_predictions))

    track_conf_matrix = confusion_matrix(track_targets, track_predictions)
    print("Matrica konfuzije za Track Task:")
    print(track_conf_matrix)

    resman_conf_matrix = confusion_matrix(resman_targets.argmax(axis=1), resman_predictions.argmax(axis=1))
    print("Matrica konfuzije za Resman Task:")
    print(resman_conf_matrix)

    sysmon_conf_matrix = confusion_matrix(sysmon_targets, sysmon_predictions)
    print("Matrica konfuzije za Sysmon Task:")
    print(sysmon_conf_matrix)
