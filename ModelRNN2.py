import os
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.metrics import confusion_matrix

# Regression

class MultiTaskHybridModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiTaskHybridModel, self).__init__()
        self.hidden_size = hidden_size

        # Shared RNN layer
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # Task 1: Track
        self.track_output_layer = nn.Linear(hidden_size, 1)  # Continuous output for track
        
        # Task 2: Resman
        self.resman_output_layer = nn.Linear(hidden_size, 2)  # Binary classification for resman
        
        # Task 3: Sysmon
        self.sysmon_fc1 = nn.Linear(input_size, hidden_size)
        self.sysmon_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        rnn_output, _ = self.rnn(inputs)
        last_output = rnn_output[:, -1, :]  # Use the last time step output
        
        # Task 1: Track
        track_output = self.track_output_layer(last_output)
        
        # Task 2: Resman
        resman_output = self.resman_output_layer(last_output)
        
        # Task 3: Sysmon
        sysmon_fc1_output = torch.relu(self.sysmon_fc1(inputs[:, -1, :]))  # Use the last input step
        sysmon_output = self.sysmon_fc2(sysmon_fc1_output)

        return track_output, resman_output, sysmon_output

class CustomDataset(Dataset):
    def __init__(self, df, input_cols, output_cols, scenario_col, sequence_length=10):
        self.df = df.reset_index(drop=True)  
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.scenario_col = scenario_col
        self.sequence_length = sequence_length
        
        self.df.columns = self.df.columns.str.replace(r', \d+', '', regex=True)
        
        missing_input_cols = [col for col in self.input_cols if col not in self.df.columns]
        if missing_input_cols:
            raise KeyError(f"Missing input columns: {missing_input_cols}")

        missing_output_cols = [col for col in self.output_cols if col not in self.df.columns]
        if missing_output_cols:
            raise KeyError(f"Missing output columns: {missing_output_cols}")

        self.df[self.input_cols] = self.df[self.input_cols].apply(pd.to_numeric, errors='coerce')
        self.df[self.input_cols] = self.df[self.input_cols].fillna(0)  

    def __len__(self):
        num_samples = len(self.df) - self.sequence_length + 1
        num_batches = math.ceil(num_samples / batch_size)
        return num_batches * batch_size 

    def __getitem__(self, idx):
        if idx + self.sequence_length > len(self.df):
            idx = len(self.df) - self.sequence_length

        inputs = self.df.loc[idx:idx+self.sequence_length-1, self.input_cols].values
        inputs = torch.tensor(inputs.astype(float), dtype=torch.float32)

        sample = self.df.iloc[idx + self.sequence_length - 1]

        cursor_in_target = sample['cursor_in_target']
        a_in_tolerance = sample['a_in_tolerance']
        b_in_tolerance = sample['b_in_tolerance']
        response_time=sample['response_time']
        

        outputs = torch.tensor([cursor_in_target, a_in_tolerance, b_in_tolerance,response_time], dtype=torch.float32)
        
        scenario = sample[self.scenario_col]

        return inputs, outputs, scenario
def evaluate_model(model, dataloader):
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
            track_predictions = track_output.cpu().numpy()
            track_targets = targets[:, 0].cpu().numpy()
            all_track_predictions.extend(track_predictions)
            all_track_targets.extend(track_targets)

            # Task 2: Resman
            resman_predictions = torch.sigmoid(resman_output).cpu().numpy()
            resman_targets = targets[:, 1:3].cpu().numpy()
            all_resman_predictions.extend(resman_predictions)
            all_resman_targets.extend(resman_targets)

            # Task 3: Sysmon
            sysmon_predictions = sysmon_output.cpu().numpy()
            sysmon_targets = targets[:, 3:4].cpu().numpy()
            all_sysmon_predictions.extend(sysmon_predictions)
            all_sysmon_targets.extend(sysmon_targets)

    return (np.array(all_track_predictions), np.array(all_track_targets),
            np.array(all_resman_predictions), np.array(all_resman_targets),
            np.array(all_sysmon_predictions), np.array(all_sysmon_targets))




def find_sequence_csv_files(root_dir):
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "sequence" in file and file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def load_and_combine_csv_files(csv_files):
    combined_df = pd.DataFrame()
    for file in csv_files:
        scenario_df = pd.read_csv(file)
        #print(f"Columns in {file}: {scenario_df.columns.tolist()}")
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
    num_epochs = 15  
    hidden_size = 64
    input_size = 19  
    batch_size = 32
    sequence_length = 10

    root_directory = 'OpenMATB/sessions/2024-08-17'  
    csv_files = find_sequence_csv_files(root_directory)
    combined_df = load_and_combine_csv_files(csv_files)

    train_df, test_df = train_test_split(combined_df, test_size=0.7, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    combined_input_cols = [
        'center_deviation', 'joystick_x', 'joystick_y',  # Track input cols
        'a_deviation', 'b_deviation', 'pump_1_flow', 'pump_2_flow', 'pump_3_flow', 'pump_4_flow', 'pump_5_flow', 'pump_6_flow', 'pump_7_flow', 'pump_8_flow',  # Resman input cols
        'scale1', 'scale2', 'scale3', 'scale4', 'event_occured', 'response_time'  # Sysmon input cols
    ]

    combined_output_cols = [
        'cursor_in_target',  # Track output cols
        'a_in_tolerance', 'b_in_tolerance',  # Resman output cols
        'response_time'  # Sysmon output cols
    ]

    scenario_col = 'scenario'  # Column containing scenario labels

    train_dataset = CustomDataset(train_df, combined_input_cols, combined_output_cols, scenario_col, sequence_length=sequence_length)
    test_dataset = CustomDataset(test_df, combined_input_cols, combined_output_cols, scenario_col, sequence_length=sequence_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scenario_weights = {'easy': 1.0, 'medium': 1.2, 'hard': 1.5}

    # Model    
    model= MultiTaskHybridModel(input_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss functions
    criterion_track = nn.MSELoss()
    criterion_resman = nn.BCEWithLogitsLoss()
    criterion_sysmon = nn.MSELoss()
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, scenario in train_dataloader:
            optimizer.zero_grad()

            track_output, resman_output, sysmon_output = model(inputs)

            track_loss = criterion_track(track_output.squeeze(), targets[:, 0])
            resman_loss = criterion_resman(resman_output, targets[:, 1:3])
            sysmon_loss = criterion_sysmon(sysmon_output.squeeze(), targets[:, 3])

            loss = track_loss + resman_loss + sysmon_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}")

    (track_predictions, track_targets,
     resman_predictions, resman_targets,
     sysmon_predictions, sysmon_targets) = evaluate_model(model, test_dataloader)

    print("Track Task - MSE:", np.mean((track_targets - track_predictions) ** 2))

    print("Resman Task - Accuracy:", accuracy_score(resman_targets, (resman_predictions > 0.5).astype(int)))
    print("Resman Task - Classification Report:")
    print(classification_report(resman_targets, (resman_predictions > 0.5).astype(int)))


    track_predictions_binary = (track_predictions > 0.5).astype(int)
    #track_conf_matrix = confusion_matrix(track_targets, track_predictions)
    print("Matrica konfuzije za Track Task:")
    #print(track_conf_matrix)

    resman_conf_matrix = confusion_matrix(resman_targets.argmax(axis=1), resman_predictions.argmax(axis=1))
    print("Matrica konfuzije za Resman Task:")
    print(resman_conf_matrix)

    print("Sysmon Task - MSE:", np.mean((sysmon_targets - sysmon_predictions) ** 2))

