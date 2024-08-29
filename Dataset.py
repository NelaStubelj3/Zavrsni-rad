import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, input_cols, output_cols, scenario_col, sequence_length=10):
        self.df = df.reset_index(drop=True)
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.scenario_col = scenario_col
        self.sequence_length = sequence_length

        # Pretvori ulaze u numeričke vrijednosti i rukuj nenumeričkim vrijednostima
        self.df[self.input_cols] = self.df[self.input_cols].apply(pd.to_numeric, errors='coerce')
        self.df[self.input_cols] = self.df[self.input_cols].fillna(0)  # ili druga strategija za rukovanje NaN vrijednostima

        # Kombiniraj HIT i MISS stupce u jednu binarnu oznaku
        self.df['sysmon_label'] = self.df.apply(lambda row: 1 if row['signal_detection_HIT'] == 1 else 0, axis=1)

    def __len__(self):
        return len(self.df) - self.sequence_length + 1

    def __getitem__(self, idx):
        if idx + self.sequence_length > len(self.df):
            idx = len(self.df) - self.sequence_length

        inputs = self.df.loc[idx:idx + self.sequence_length - 1, self.input_cols].values
        inputs = torch.tensor(inputs.astype(float), dtype=torch.float32)

        sample = self.df.iloc[idx + self.sequence_length - 1]

        cursor_in_target = sample['cursor_in_target']
        a_in_tolerance = sample['a_in_tolerance']
        b_in_tolerance = sample['b_in_tolerance']
        sysmon_label = sample['sysmon_label']

        outputs = torch.tensor([cursor_in_target, a_in_tolerance, b_in_tolerance, sysmon_label], dtype=torch.float32)
        
        scenario = sample[self.scenario_col]

        return inputs, outputs, scenario

# Definiraj stupce
combined_input_cols = [
    'center_deviation', 'joystick_x', 'joystick_y',  # Track ulazni stupci
    'a_deviation', 'b_deviation', 'pump_1_flow', 'pump_2_flow', 'pump_3_flow', 'pump_4_flow', 'pump_5_flow', 'pump_6_flow', 'pump_7_flow', 'pump_8_flow',  # Resman ulazni stupci
    'scale1', 'scale2', 'scale3', 'scale4', 'event_occured', 'reacted'  # Sysmon ulazni stupci
]

combined_output_cols = [
    'cursor_in_target',  # Track izlazni stupac
    'a_in_tolerance', 'b_in_tolerance',  # Resman izlazni stupci
    'sysmon_label'  # Sysmon izlazni stupac (binarna oznaka)
]

scenario_col = 'scenario'  # Stupac koji sadrži oznake scenarija

# Učitaj DataFrame
df = pd.read_csv('OpenMATB/sessions/2024-06-05/Session 66 hard/66_240605_180052_sequence.csv')

# Kreiraj instancu skupa podataka
dataset = CustomDataset(df, combined_input_cols, combined_output_cols, scenario_col)
import torch.nn as nn

class MultiTaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiTaskRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN zajednički sloj
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # Zadatak 1: Track
        self.track_output_layer = nn.Linear(hidden_size, 1)  # cursor_in_target
        
        # Zadatak 2: Resman
        self.resman_output_layer = nn.Linear(hidden_size, 2)  # a_in_tolerance, b_in_tolerance
        
        # Zadatak 3: Sysmon
        self.sysmon_output_layer = nn.Linear(hidden_size, 2)  # binarna klasifikacija za signal_detection (HIT ili MISS)

    def forward(self, inputs):
        rnn_output, _ = self.rnn(inputs)
        last_output = rnn_output[:, -1, :]  # Koristi izlaz posljednjeg vremenskog koraka
        
        # Zadatak 1: Track
        track_output = self.track_output_layer(last_output)
        
        # Zadatak 2: Resman
        resman_output = self.resman_output_layer(last_output)
        
        # Zadatak 3: Sysmon
        sysmon_output = self.sysmon_output_layer(last_output)

        return track_output, resman_output, sysmon_output

def train(model, dataloader, num_epochs=10):
    criterion_track = nn.BCEWithLogitsLoss()
    criterion_resman = nn.BCEWithLogitsLoss()
    criterion_sysmon = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets, _ in dataloader:
            track_target = targets[:, 0].unsqueeze(1)
            resman_target = targets[:, 1:3]
            sysmon_target = targets[:, 3].long()  # Osiguraj da ciljana oznaka za Sysmon bude tipa long za CrossEntropyLoss

            optimizer.zero_grad()

            track_output, resman_output, sysmon_output = model(inputs)

            loss_track = criterion_track(track_output, track_target)
            loss_resman = criterion_resman(resman_output, resman_target)
            loss_sysmon = criterion_sysmon(sysmon_output, sysmon_target)

            total_loss = loss_track + loss_resman + loss_sysmon
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

if __name__ == "__main__":
    input_size = 19  # Ukupna veličina ulaza za kombinirane zadatke
    hidden_size = 64

    model = MultiTaskRNN(input_size, hidden_size)
    
    # Definiraj DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    train(model, dataloader, num_epochs=10)
