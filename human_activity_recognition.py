# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
roeiyehezkel_dlw2test_path = kagglehub.dataset_download('roeiyehezkel/dlw2test')
print('Data source import complete.')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os

os.environ['KAGGLE_USERNAME'] = 'roeiyehezkel'
os.environ['KAGGLE_KEY'] = '648e50e7a636536870f2b596a2b0afc0'


!kaggle competitions download -c bgu-i-cant-see-you-but-you-are-reading-a-book


import zipfile
import os

# Path to ZIP file
zip_path = "bgu-i-cant-see-you-but-you-are-reading-a-book.zip"
extract_dir = "/kaggle/temp/data"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    file_list = zf.namelist()
    for file in file_list:
        zf.extract(file, path=extract_dir)
print(f"Extracted {len(file_list)} files to {extract_dir}")


!pip install mlflow

### Core Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import mlflow.sklearn
import os
import seaborn as sns

# Torch and PyTorch Lightning
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import LightningDataModule
import torch.optim as optim
import torch.nn.functional as F

# Scikit-learn Libraries
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss,classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# MLflow for Logging
import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import Callback





# Load the train.csv dataset
train = pd.read_csv('/kaggle/temp/data/train.csv')

# Plot the activity distribution
plt.figure(figsize=(10, 6))
train['activity'].value_counts().plot(kind='bar')
plt.title('Activity Distribution')
plt.xlabel('Activity')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# Load a sample file (type #1 or type #2)
sample_data = pd.read_csv('/kaggle/temp/data/unlabeled/unlabeled/1.csv')

# Filter for acceleration data (type #1 only)
acceleration_data = sample_data[sample_data['measurement type'] == 'acceleration [m/s/s]']

# Plot acceleration data
plt.figure(figsize=(12, 6))
plt.plot(acceleration_data['x'], label='X-axis')
plt.plot(acceleration_data['y'], label='Y-axis')
plt.plot(acceleration_data['z'], label='Z-axis')
plt.title('Sample Acceleration Data')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()


# Aggregate and visualize acceleration data across activities
plt.figure(figsize=(10, 6))
train['userid'].value_counts().plot(kind='bar')
plt.title('Number of Records per User')
plt.xlabel('User ID')
plt.ylabel('Count')
plt.show()

print(train['activity'].value_counts())


# Group by 'activity' and 'userid' and count occurrences
activity_user_df = train.groupby(['activity', 'userid']).size().unstack(fill_value=0)
# Get the list of unique user IDs
user_ids = train['userid'].unique()
for user in user_ids:
    user_data = train[train['userid'] == user]

    # Print the activity distribution for the user
    print(f"Activity Distribution for {user}:")
    print(user_data['activity'].value_counts())
    print("-" * 50)

    # Plot the activity distribution for the user
    plt.figure(figsize=(10, 6))
    user_data['activity'].value_counts().plot(kind='bar')
    plt.title(f'Activity Distribution for {user}')
    plt.xlabel('Activity')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

activity_user_df

#maximal and minimal values of sequence_length from train df per sensor(smartwatch or vicon):
print(train.groupby('sensor')['sequence_length'].max())
print(train.groupby('sensor')['sequence_length'].min())


train = pd.read_csv('/kaggle/temp/data/train.csv')
train['userid'] = train['userid'].apply(lambda x: int(x[-2:]))
train = pd.get_dummies(train, columns=['sensor', 'body_part', 'side'])
for col in ['sensor_smartwatch','sensor_vicon', 'body_part_hand','body_part_foot', 'side_right', 'side_left']:
    train[col] = train[col].astype(int)
print(train['sequence_length'])
max_len = max(list(train['sequence_length']))
print(max_len)
# train.drop(columns=['sequence_length'])
encoder = LabelEncoder()
train['activity_enc'] = encoder.fit_transform(train['activity'])
labels_dict = {row[1]['id'] : row[1]['activity_enc'] for row in train.iterrows()}



# Function to extract statistical features from a sequence
def extract_statistical_features(sequence):
    features = {}
    for col in ["x", "y", "z"]:
        features[f"{col}_mean"] = sequence[col].mean()
        features[f"{col}_std"] = sequence[col].std()
        features[f"{col}_skew"] = skew(sequence[col])
        features[f"{col}_kurtosis"] = kurtosis(sequence[col])
    return features

# Function to calculate Signal Magnitude Area (SMA)
def calculate_sma(sequence):
    sma = np.sum(np.abs(sequence[["x", "y", "z"]]), axis=1).mean()
    return {"sma": sma}

# Function to calculate zero-crossing rate
def calculate_zero_crossing_rate(sequence):
    zcr = {}
    for col in ["x", "y", "z"]:
        zcr[f"{col}_zcr"] = ((sequence[col][1:].values * sequence[col][:-1].values) < 0).mean()
    return zcr

# Function to process sensor data and unify features
def process_sensor_data(sequence, metadata_row):
    # Extract statistical features
    stats_features = extract_statistical_features(sequence)

    # Calculate SMA
    sma_features = calculate_sma(sequence)

    # Calculate zero-crossing rate
    zcr_features = calculate_zero_crossing_rate(sequence)

    # Combine all features
    features = {**stats_features, **sma_features, **zcr_features}

    # Add metadata features (e.g., right vs. left, hand vs. foot)
    for col in ["sensor_smartwatch", "sensor_vicon", "body_part_hand", "body_part_foot", "side_right", "side_left"]:
        features[col] = metadata_row[col]

    # Add sequence length as a feature
    features["sequence_length"] = len(sequence)

    return features

# Main function to process all sequences
def process_sequences(data, dfs):
    feature_list = []
    labels = []

    for ID in data["id"].values:
        if ID in dfs:
            sequence = dfs[ID]
            metadata_row = data[data["id"] == ID].iloc[0]

            # Process the sequence and extract features
            features = process_sensor_data(sequence, metadata_row)
            feature_list.append(features)

            # Add the corresponding label
            labels.append(metadata_row["activity"])

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list)
    return feature_df, labels

# Preprocess the data directory
directory = '/kaggle/temp/data/unlabeled/unlabeled/'
columns = ['sensor_smartwatch', 'sensor_vicon', 'body_part_hand', 'body_part_foot', 'side_right', 'side_left']
dfs = {}
train_ids = train['id'].values  # Get all train IDs

for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        file_id = int(filename[:-4])  # Extract the file ID from the filename
        type = 0

        # Process only files where file_id is in train_ids
        if file_id in train_ids:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'measurement type' column exists
                if 'measurement type' in df.columns:
                    type = 1
                    # Filter for acceleration data
                    df = df[df['measurement type'] == 'acceleration [m/s/s]']

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

                    # Drop the 'measurement type' column
                    df.drop(columns=['measurement type'], inplace=True)
                else:
                    type = 2
                    # Rename columns for location data
                    df.rename(columns={
                        'x [m]': 'x',
                        'y [m]': 'y',
                        'z [m]': 'z'
                    }, inplace=True)

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

                # Store the DataFrame in the dictionary
                dfs[file_id] = df

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Check how many files were loaded
print(f"Loaded {len(dfs)} files.")

# Perform a regular train-test split
train_set, val_set = train_test_split(
    train,
    test_size=0.2,  # Adjust test size as needed (e.g., 20% for validation)
    stratify=train['activity'],  # Ensure label distribution is maintained
    random_state=42  # For reproducibility
)

# Process train and validation sequences
train_features, train_labels = process_sequences(train_set, dfs)
val_features, val_labels = process_sequences(val_set, dfs)

from sklearn.metrics import log_loss
import numpy as np

# Calculate class distribution in the training set
class_counts = train_set['activity'].value_counts()
class_probabilities = class_counts / class_counts.sum()

# Convert to a dictionary for easier mapping
class_distribution = class_probabilities.to_dict()

# Create a DataFrame with predicted probabilities for the validation set
val_predictions = pd.DataFrame(
    np.tile(list(class_distribution.values()), (len(val_set), 1)),
    columns=class_distribution.keys()
)

# Encode true labels for log-loss calculation
label_encoder = LabelEncoder()
train_set['activity_enc'] = label_encoder.fit_transform(train_set['activity'])
val_set['activity_enc'] = label_encoder.transform(val_set['activity'])

# Calculate log-loss for the validation set
val_logloss = log_loss(val_set['activity_enc'], val_predictions)

print(f"Validation Log-Loss for Naïve Baseline: {val_logloss:.4f}")


train_features.columns

# Start an MLflow run
with mlflow.start_run():
    # Ensure labels are encoded
    label_encoder = LabelEncoder()
    train_encoded_labels = label_encoder.fit_transform(train_labels)
    val_encoded_labels = label_encoder.transform(val_labels)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(train_features, train_encoded_labels)

    # Predictions and Metrics
    train_probabilities = rf_model.predict_proba(train_features)
    val_probabilities = rf_model.predict_proba(val_features)

    train_logloss = log_loss(train_encoded_labels, train_probabilities)
    val_logloss = log_loss(val_encoded_labels, val_probabilities)

    train_predictions = rf_model.predict(train_features)
    val_predictions = rf_model.predict(val_features)

    train_accuracy = accuracy_score(train_encoded_labels, train_predictions)
    val_accuracy = accuracy_score(val_encoded_labels, val_predictions)

    # Log parameters
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("train_log_loss", train_logloss)
    mlflow.log_metric("val_log_loss", val_logloss)

    # Log classification report as an artifact
    val_classification_report = classification_report(val_encoded_labels, val_predictions)
    with open("/kaggle/working/classification_report.txt", "w") as f:
        f.write(val_classification_report)
    mlflow.log_artifact("/kaggle/working/classification_report.txt")

    # Log the trained model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

# Print metrics to console
print(f"Validation Accuracy (Random Forest): {val_accuracy:.4f}")
print(f"Training Log Loss: {train_logloss:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(val_encoded_labels, val_predictions)

# Get label names from the label encoder
label_names = label_encoder.classes_

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import torch
import mlflow.sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Function to extract statistical features from a sequence
def extract_statistical_features(sequence):
    features = {}
    for col in ["x", "y", "z"]:
        features[f"{col}_mean"] = sequence[col].mean()
        features[f"{col}_std"] = sequence[col].std()
        features[f"{col}_skew"] = skew(sequence[col])
        features[f"{col}_kurtosis"] = kurtosis(sequence[col])
    return features

# Function to calculate Signal Magnitude Area (SMA)
def calculate_sma(sequence):
    sma = np.sum(np.abs(sequence[["x", "y", "z"]]), axis=1).mean()
    return {"sma": sma}

# Function to calculate zero-crossing rate
def calculate_zero_crossing_rate(sequence):
    zcr = {}
    for col in ["x", "y", "z"]:
        zcr[f"{col}_zcr"] = ((sequence[col][1:].values * sequence[col][:-1].values) < 0).mean()
    return zcr

# Function to process sensor data and unify features
def process_sensor_data(sequence):
    # Extract statistical features
    stats_features = extract_statistical_features(sequence)

    # Calculate SMA
    sma_features = calculate_sma(sequence)

    # Calculate zero-crossing rate
    zcr_features = calculate_zero_crossing_rate(sequence)

    # Combine all features
    features = {**stats_features, **sma_features, **zcr_features}

    # # Add metadata features (e.g., right vs. left, hand vs. foot)
    # for col in ["sensor_smartwatch", "sensor_vicon", "body_part_hand", "body_part_foot", "side_right", "side_left"]:
    #     features[col] = metadata_row[col]

    # Add sequence length as a feature
    # features["sequence_length"] = len(sequence)

    return features

# Main function to process all sequences
def process_sequences(data, filename, id_list, directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        file_id = int(filename[:-4])  # Extract the file ID from the filename
        type = 0
        # Process only files where file_id is in train_ids
        if file_id in id_list:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'measurement type' column exists
                if 'measurement type' in df.columns:
                    type = 1
                    # Filter for acceleration data
                    df = df[df['measurement type'] == 'acceleration [m/s/s]']

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

                    # Drop the 'measurement type' column
                    df.drop(columns=['measurement type'], inplace=True)
                else:
                    type = 2
                    # Rename columns for location data
                    df.rename(columns={
                        'x [m]': 'x',
                        'y [m]': 'y',
                        'z [m]': 'z'
                    }, inplace=True)

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                return None

    # Process the sequence and extract features
    features = process_sensor_data(df)
    for col in features.keys():
        df[col] = features[col]
    # Add the corresponding label
    label = labels_dict[file_id]
    seq = torch.tensor(df.values, dtype=torch.float32).permute(1, 0)

    return seq, label

# Preprocess the data directory
directory = '/kaggle/temp/data/unlabeled/unlabeled/'
columns = ['sensor_smartwatch', 'sensor_vicon', 'body_part_hand', 'body_part_foot', 'side_right', 'side_left']


# Perform a regular train-test split
train_set, val_set = train_test_split(
    train,
    test_size=0.2,  # Adjust test size as needed (e.g., 20% for validation)
    stratify=train['activity'],  # Ensure label distribution is maintained
    random_state=42  # For reproducibility
)

train_ids = list(train_set['id'])
val_ids = list(val_set['id'])


def pad_to_length(sequence, target_length, padding_value=0.0):
        """
        Pads a sequence to a specific target length.
        Args:
            sequence (Tensor): The input tensor sequence of shape (sequence_length, features).
            target_length (int): The target length to pad the sequence to.
            padding_value (float): The value to pad with (default is 0.0).
        Returns:
            Tensor: The padded sequence.
        """
        seq_length = sequence.size(1)
        if seq_length < target_length:
            # Calculate padding
            padding_needed = target_length - seq_length
            # Pad at the end of the sequence (padding_value is the value to use)
            padded_sequence = F.pad(sequence, (0, padding_needed), value=padding_value)
        else:
            # If the sequence is already long enough, no padding is needed
            padded_sequence = sequence[:target_length]  # Optionally truncate if it's too long
        return padded_sequence

class ProcessedSequenceDataset(Dataset):
    def __init__(self, data_dir, labels, id_list, max_len, padding_value=0.0):
        """
        Dataset that preprocesses features (scaling and padding) in the initializer.
        Args:
        - features: List of DataFrames (one per sequence).
        - labels: List of corresponding labels for the sequences.
        - padding_value: Value to use for padding.
        """
        #convert features to tensors
        self.files = [f'{ID}.csv' for ID in id_list]
        self.labels = labels
        self.padding_value = padding_value
        self.id_list = id_list
        self.data_dir = data_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        """
        Get a single sequence and its label.
        """
        filename = self.files[idx]
        seq, label =  process_sequences(train, filename, self.id_list, self.data_dir)
        pad_seq = pad_to_length(seq, self.max_len, self.padding_value)
        return pad_seq, label

data_dir = '/kaggle/temp/data/unlabeled/unlabeled/'

train_dataset = ProcessedSequenceDataset(data_dir,labels_dict, train_ids,max_len)
val_dataset = ProcessedSequenceDataset(data_dir,labels_dict,val_ids, max_len)

class SequenceDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)


class CNN1DClassifier(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate=0.001, dropout_rate=0.3, weight_decay=1e-4):
        super(CNN1DClassifier, self).__init__()
        self.save_hyperparameters()

        # Define the model
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        mlflow.log_metric("val_loss", loss.item(), step=self.global_step)
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        mlflow.log_metric("val_acc", acc.item(), step=self.global_step)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


val_loader = data_module.val_dataloader()
class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

# Setup MLflow
mlflow.set_experiment("CNN_Experiment")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="CNN_Experiment",
    run_name="CNN_1"
)



# Define input dimensions and parameters
max_len = train_dataset.max_len  # Adjust based on dataset
input_dim = max_len  # Number of features (e.g., x, y, z axes)
hidden_dim = 128
num_classes = len(set(labels_dict.values()))  # Number of unique labels
learning_rate = 0.0005

# Initialize Model
# Hyperparameters
input_channels = 25  # Replace with the actual number of input channels
num_classes = 18
batch_size = 64
learning_rate = 0.001
max_epochs = 10

# Model
model = CNN1DClassifier(input_channels=input_channels, num_classes=num_classes, learning_rate=learning_rate)

# DataModule
data_module = SequenceDataModule(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size)

# trainer = Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs)

# Trainer setup
# Initialize the metrics tracker
metrics_tracker = MetricsTracker()

# Trainer setup
trainer = pl.Trainer(
    max_epochs=25,
    logger=mlflow_logger,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model'),
        metrics_tracker  # Add metrics tracker callback here
    ],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)

val_loader = data_module.val_dataloader()

with mlflow.start_run():


    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log validation results
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

# Save metrics to file
metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")


# Visualizations
# Plot training and validation loss
# metrics = trainer.logged_metrics
# train_losses = metrics["train_loss"]
# val_losses = metrics["val_loss"]
train_losses = metrics_tracker.train_losses
validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
# plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(metrics_tracker.val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Function to extract predictions from the model
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]
def extract_predictions(model, dataloader, activity_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
# Extract predictions, true labels, and probabilities
predictions, true_labels, probabilities = extract_predictions(model, val_loader, activity_classes)

# High confidence threshold (e.g., >90%)
high_confidence_threshold = 0.9
# Low confidence threshold (e.g., <60%)
low_confidence_threshold = 0.6

# Identify good classifications (correct predictions with high confidence)
good_indices = [
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify bad classifications (incorrect predictions with high confidence)
bad_indices = [
    i for i in range(len(predictions))
    if predictions[i] != true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify uncertain predictions (confidence below threshold)
uncertain_indices = [
    i for i in range(len(predictions))
    if max(probabilities[i]) < low_confidence_threshold
]

# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
    idx)

# Function to extract statistical features from a sequence
def extract_statistical_features(sequence):
    features = {}
    for col in ["x", "y", "z"]:
        features[f"{col}_mean"] = sequence[col].mean()
        features[f"{col}_std"] = sequence[col].std()
        features[f"{col}_skew"] = skew(sequence[col])
        features[f"{col}_kurtosis"] = kurtosis(sequence[col])
    return features

# Function to calculate Signal Magnitude Area (SMA)
def calculate_sma(sequence):
    sma = np.sum(np.abs(sequence[["x", "y", "z"]]), axis=1).mean()
    return {"sma": sma}

# Function to calculate zero-crossing rate
def calculate_zero_crossing_rate(sequence):
    zcr = {}
    for col in ["x", "y", "z"]:
        zcr[f"{col}_zcr"] = ((sequence[col][1:].values * sequence[col][:-1].values) < 0).mean()
    return zcr

# Function to process sensor data and unify features
def process_sensor_data(sequence):
    # Extract statistical features
    stats_features = extract_statistical_features(sequence)

    # Calculate SMA
    sma_features = calculate_sma(sequence)

    # Calculate zero-crossing rate
    zcr_features = calculate_zero_crossing_rate(sequence)

    # Combine all features
    features = {**stats_features, **sma_features, **zcr_features}

    # # Add metadata features (e.g., right vs. left, hand vs. foot)
    # for col in ["sensor_smartwatch", "sensor_vicon", "body_part_hand", "body_part_foot", "side_right", "side_left"]:
    #     features[col] = metadata_row[col]

    # Add sequence length as a feature
    # features["sequence_length"] = len(sequence)

    return features

# Main function to process all sequences
def process_sequences(data, filename, id_list, directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        file_id = int(filename[:-4])  # Extract the file ID from the filename
        type = 0
        # Process only files where file_id is in train_ids
        if file_id in id_list:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'measurement type' column exists
                if 'measurement type' in df.columns:
                    type = 1
                    # Filter for acceleration data
                    df = df[df['measurement type'] == 'acceleration [m/s/s]']

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

                    # Drop the 'measurement type' column
                    df.drop(columns=['measurement type'], inplace=True)
                else:
                    type = 2
                    # Rename columns for location data
                    df.rename(columns={
                        'x [m]': 'x',
                        'y [m]': 'y',
                        'z [m]': 'z'
                    }, inplace=True)

                    # Add relevant columns based on train DataFrame
                    for col in columns:
                        val = train[train['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                return None

    # Process the sequence and extract features
    features = process_sensor_data(df)
    for col in features.keys():
        df[col] = features[col]
    # Add the corresponding label
    label = labels_dict[file_id]
    seq = torch.tensor(df.values, dtype=torch.float32).permute(1, 0)

    return seq, label

# Preprocess the data directory
directory = '/kaggle/temp/data/unlabeled/unlabeled/'
columns = ['sensor_smartwatch', 'sensor_vicon', 'body_part_hand', 'body_part_foot', 'side_right', 'side_left']


# Extract unique user IDs
user_ids = train['userid'].unique()

# Split user IDs into training and validation sets
train_user_ids, val_user_ids = train_test_split(
    user_ids, test_size=0.2, random_state=42  # Adjust the split ratio as needed
)

# Create training and validation sets based on user IDs
val_user_ids = [4, 5]

# Split the dataset
train_set = train[~train['userid'].isin(val_user_ids)]  # Users not in test_user_ids
val_set = train[train['userid'].isin(val_user_ids)]   # Users in test_user_ids

# Extract IDs for sequences
train_ids = list(train_set['id'])
test_ids = list(val_set['id'])

train_set = train_set.drop(columns=['userid'])
val_set = val_set.drop(columns=['userid'])


def pad_to_length(sequence, target_length, padding_value=0.0):
        """
        Pads a sequence to a specific target length.
        Args:
            sequence (Tensor): The input tensor sequence of shape (sequence_length, features).
            target_length (int): The target length to pad the sequence to.
            padding_value (float): The value to pad with (default is 0.0).
        Returns:
            Tensor: The padded sequence.
        """
        seq_length = sequence.size(1)
        if seq_length < target_length:
            # Calculate padding
            padding_needed = target_length - seq_length
            # Pad at the end of the sequence (padding_value is the value to use)
            padded_sequence = F.pad(sequence, (0, padding_needed), value=padding_value)
        else:
            # If the sequence is already long enough, no padding is needed
            padded_sequence = sequence[:target_length]  # Optionally truncate if it's too long
        return padded_sequence



class ProcessedSequenceDataset(Dataset):
    def __init__(self, data_dir, labels, id_list, max_len, padding_value=0.0):
        """
        Dataset that preprocesses features (scaling and padding) in the initializer.
        Args:
        - features: List of DataFrames (one per sequence).
        - labels: List of corresponding labels for the sequences.
        - padding_value: Value to use for padding.
        """
        #convert features to tensors
        self.files = [f'{ID}.csv' for ID in id_list]
        self.labels = labels
        self.padding_value = padding_value
        self.id_list = id_list
        self.data_dir = data_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        """
        Get a single sequence and its label.
        """
        filename = self.files[idx]
        seq, label =  process_sequences(train, filename, self.id_list, self.data_dir)
        pad_seq = pad_to_length(seq, self.max_len, self.padding_value)
        return pad_seq, label

    def collate_fn(batch):
        sequences, labels = zip(*batch)

        # Pad sequences along the time dimension (dim=0)
        padded_sequences = pad_sequence(sequences, batch_first=True)

        labels = torch.tensor(labels)
        res = (padded_sequences, labels)
        return res




data_dir = '/kaggle/temp/data/unlabeled/unlabeled/'

train_dataset = ProcessedSequenceDataset(data_dir,labels_dict, train_ids,max_len)
val_dataset = ProcessedSequenceDataset(data_dir,labels_dict,val_ids, max_len)
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ProcessedSequenceDataset.collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ProcessedSequenceDataset.collate_fn, num_workers=4)



class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

# Setup MLflow
mlflow.set_experiment("Simple_LSTM_Experiment")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="Simple_LSTM_Experiment",
    run_name="LSTM_with_BatchNorm"
)



# LSTM Model with Batch Normalization
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, bidirectional=True, learning_rate=0.001):
        super(SimpleLSTM, self).__init__()
        self.save_hyperparameters()

        self.lstm_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim * (2 if bidirectional else 1)
            lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)
            self.batch_norm_layers.append(
                nn.BatchNorm1d(hidden_dim * (2 if bidirectional else 1))
            )

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        for lstm_layer, batch_norm_layer in zip(self.lstm_layers, self.batch_norm_layers):
            x, _ = lstm_layer(x)
            x = x.permute(0, 2, 1)  # Switch to (batch, features, sequence_length)
            x = batch_norm_layer(x)
            x = x.permute(0, 2, 1)  # Switch back to (batch, sequence_length, features)

        x = x[:, -1, :]  # Last time step
        output = self.fc(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Define input dimensions and parameters
max_len = train_loader.dataset.max_len  # Adjust based on dataset
input_dim = max_len  # Number of features (e.g., x, y, z axes)
hidden_dim = 128
num_classes = len(set(labels_dict.values()))  # Number of unique labels
learning_rate = 0.0005

# Initialize Model
model = SimpleLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=3,
    bidirectional=False,
    learning_rate=learning_rate
)

# Trainer setup
# Initialize the metrics tracker
metrics_tracker = MetricsTracker()

# Trainer setup
trainer = pl.Trainer(
    max_epochs=10,
    precision=16,
    logger=mlflow_logger,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model'),
        metrics_tracker  # Add metrics tracker callback here
    ],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)


# Train and Validate the Model
with mlflow.start_run():
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log validation results
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

# Save metrics to file
metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")





train_losses = metrics_tracker.train_losses
validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(metrics_tracker.val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



# Function to extract predictions from the model
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]
def extract_predictions(model, dataloader, activity_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
# Extract predictions, true labels, and probabilities
predictions, true_labels, probabilities = extract_predictions(model, val_loader, activity_classes)

# High confidence threshold (e.g., >90%)
high_confidence_threshold = 0.9
# Low confidence threshold (e.g., <60%)
low_confidence_threshold = 0.6

# Identify good classifications (correct predictions with high confidence)
good_indices = [
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify bad classifications (incorrect predictions with high confidence)
bad_indices = [
    i for i in range(len(predictions))
    if predictions[i] != true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify uncertain predictions (confidence below threshold)
uncertain_indices = [
    i for i in range(len(predictions))
    if max(probabilities[i]) < low_confidence_threshold
]


# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
        idx
    )




# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Get label names from the label encoder
label_names = label_encoder.classes_

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()



class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

# Setup MLflow
mlflow.set_experiment("Simple_LSTM_Experiment")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="Simple_LSTM_Experiment",
    run_name="LSTM_with_BatchNorm"
)



# LSTM Model with Batch Normalization
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, bidirectional=True, learning_rate=0.001):
        super(SimpleLSTM, self).__init__()
        self.save_hyperparameters()

        self.lstm_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim * (2 if bidirectional else 1)
            lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)
            self.batch_norm_layers.append(
                nn.BatchNorm1d(hidden_dim * (2 if bidirectional else 1))
            )

        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        for lstm_layer, batch_norm_layer in zip(self.lstm_layers, self.batch_norm_layers):
            x, _ = lstm_layer(x)
            x = x.permute(0, 2, 1)  # Switch to (batch, features, sequence_length)
            x = batch_norm_layer(x)
            x = x.permute(0, 2, 1)  # Switch back to (batch, sequence_length, features)

        x = x[:, -1, :]  # Last time step
        output = self.fc(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Define input dimensions and parameters
max_len = train_loader.dataset.max_len  # Adjust based on dataset
input_dim = max_len  # Number of features (e.g., x, y, z axes)
hidden_dim = 128
num_classes = len(set(labels_dict.values()))  # Number of unique labels
learning_rate = 0.0005

# Initialize Model
model = SimpleLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=3,
    bidirectional=False,
    learning_rate=learning_rate
)

# Trainer setup
# Initialize the metrics tracker
metrics_tracker = MetricsTracker()

# Trainer setup
trainer = pl.Trainer(
    max_epochs=10,
    precision=16,
    logger=mlflow_logger,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model'),
        metrics_tracker  # Add metrics tracker callback here
    ],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)


# Train and Validate the Model
with mlflow.start_run():
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log validation results
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

# Save metrics to file
metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")





train_losses = metrics_tracker.train_losses
validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(metrics_tracker.val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



# Function to extract predictions from the model
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]
def extract_predictions(model, dataloader, activity_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
# Extract predictions, true labels, and probabilities
predictions, true_labels, probabilities = extract_predictions(model, val_loader, activity_classes)

# High confidence threshold (e.g., >90%)
high_confidence_threshold = 0.9
# Low confidence threshold (e.g., <60%)
low_confidence_threshold = 0.6

# Identify good classifications (correct predictions with high confidence)
good_indices = [
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify bad classifications (incorrect predictions with high confidence)
bad_indices = [
    i for i in range(len(predictions))
    if predictions[i] != true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify uncertain predictions (confidence below threshold)
uncertain_indices = [
    i for i in range(len(predictions))
    if max(probabilities[i]) < low_confidence_threshold
]


# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
        idx
    )



class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

# Setup MLflow
mlflow.set_experiment("Simple_LSTM_Experiment")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="Simple_LSTM_Experiment",
    run_name="LSTM_with_BatchNorm"
)



# LSTM Model with Batch Normalization
class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, num_classes, learning_rate=0.001, dropout_rate=0.3):
        """
        Simplified LSTM model with two layers and dropout.
        Args:
            input_dim (int): Input dimension (number of features).
            hidden_dims (list): List of hidden dimensions for LSTM layers (e.g., [128, 64]).
            num_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            dropout_rate (float): Dropout rate to prevent overfitting.
        """
        super(SimpleLSTM, self).__init__()
        self.save_hyperparameters()

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=1,
            batch_first=True
        )

        # Dropout between layers
        self.dropout = nn.Dropout(dropout_rate)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            num_layers=1,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dims[1], num_classes)

        # Store the learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        """
        x, _ = self.lstm1(x)  # Pass through the first LSTM layer
        x = self.dropout(x)   # Apply dropout
        x, _ = self.lstm2(x)  # Pass through the second LSTM layer

        x = x[:, -1, :]  # Take the output from the last time step
        output = self.fc(x)  # Fully connected layer for predictions
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# Define input dimensions and parameters
max_len = train_loader.dataset.max_len  # Adjust based on dataset
input_dim = max_len  # Number of features (e.g., x, y, z axes)
num_classes = len(set(labels_dict.values()))  # Number of unique labels
learning_rate = 0.0005

# Initialize the simplified LSTM model
hidden_dims = [128, 64]  # Hidden dimensions for LSTM layers
dropout_rate = 0.3       # Dropout rate to prevent overfitting

model = SimpleLSTM(
    input_dim=max_len,       # Number of input features (e.g., x, y, z)
    hidden_dims=hidden_dims, # Hidden dimensions [128, 64]
    num_classes=len(set(labels_dict.values())), # Number of classes
    learning_rate=0.0005,    # Learning rate
    dropout_rate=dropout_rate
)


# Trainer setup
# Initialize the metrics tracker
metrics_tracker = MetricsTracker()

# Trainer setup
trainer = pl.Trainer(
    max_epochs=10,
    precision=16,
    logger=mlflow_logger,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model'),
        metrics_tracker  # Add metrics tracker callback here
    ],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)


# Train and Validate the Model
with mlflow.start_run():
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dims", hidden_dims)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log validation results
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

# Save metrics to file
metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")





train_losses = metrics_tracker.train_losses
validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(metrics_tracker.val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(metrics_tracker.val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
        idx
    )


# Preprocess the test directory
directory = '/kaggle/temp/data/unlabeled/unlabeled/'
columns = ['sensor_smartwatch', 'sensor_vicon', 'body_part_hand', 'body_part_foot', 'side_right', 'side_left']
test_dfs = {}
metadata_path = '/kaggle/input/dlw2test/metadata.csv'  # Path to test metadata
test_metadata = pd.read_csv(metadata_path)
test_metadata = pd.get_dummies(test_metadata, columns=['sensor', 'body_part', 'side'])
for col in ['sensor_smartwatch','sensor_vicon', 'body_part_hand','body_part_foot', 'side_right', 'side_left']:
    test_metadata[col] = test_metadata[col].astype(int)
test_metadata.rename(columns={"sample_id": "id"}, inplace=True)

test_metadata=test_metadata.drop(columns=['userid'])
user_test_ids = test_metadata['id'].values


# Function to extract statistical features from a sequence
def extract_statistical_features(sequence):
    features = {}
    for col in ["x", "y", "z"]:
        features[f"{col}_mean"] = sequence[col].mean()
        features[f"{col}_std"] = sequence[col].std()
        features[f"{col}_skew"] = skew(sequence[col])
        features[f"{col}_kurtosis"] = kurtosis(sequence[col])
    return features

# Function to calculate Signal Magnitude Area (SMA)
def calculate_sma(sequence):
    sma = np.sum(np.abs(sequence[["x", "y", "z"]]), axis=1).mean()
    return {"sma": sma}

# Function to calculate zero-crossing rate
def calculate_zero_crossing_rate(sequence):
    zcr = {}
    for col in ["x", "y", "z"]:
        zcr[f"{col}_zcr"] = ((sequence[col][1:].values * sequence[col][:-1].values) < 0).mean()
    return zcr

# Function to process sensor data and unify features
def process_sensor_data(sequence):
    # Extract statistical features
    stats_features = extract_statistical_features(sequence)

    # Calculate SMA
    sma_features = calculate_sma(sequence)

    # Calculate zero-crossing rate
    zcr_features = calculate_zero_crossing_rate(sequence)

    # Combine all features
    features = {**stats_features, **sma_features, **zcr_features}

    # # Add metadata features (e.g., right vs. left, hand vs. foot)
    # for col in ["sensor_smartwatch", "sensor_vicon", "body_part_hand", "body_part_foot", "side_right", "side_left"]:
    #     features[col] = metadata_row[col]

    # Add sequence length as a feature
    # features["sequence_length"] = len(sequence)

    return features

# Main function to process all sequences
def process_sequences(data, filename, id_list, directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        file_id = int(filename[:-4])  # Extract the file ID from the filename
        type = 0
        # Process only files where file_id is in test_metadata_ids
        if file_id in id_list:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'measurement type' column exists
                if 'measurement type' in df.columns:
                    type = 1
                    # Filter for acceleration data
                    df = df[df['measurement type'] == 'acceleration [m/s/s]']

                    # Add relevant columns based on test_metadata DataFrame
                    for col in columns:
                        val = test_metadata[test_metadata['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

                    # Drop the 'measurement type' column
                    df.drop(columns=['measurement type'], inplace=True)
                else:
                    type = 2
                    # Rename columns for location data
                    df.rename(columns={
                        'x [m]': 'x',
                        'y [m]': 'y',
                        'z [m]': 'z'
                    }, inplace=True)

                    # Add relevant columns based on test_metadata DataFrame
                    for col in columns:
                        val = test_metadata[test_metadata['id'] == file_id][col].values
                        if len(val) > 0:
                            df[col] = val[0]

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                return None

    # Process the sequence and extract features
    features = process_sensor_data(df)
    for col in features.keys():
        df[col] = features[col]

    # Return sequence and file_id (no label for test data)
    seq = torch.tensor(df.values, dtype=torch.float32).permute(1, 0)
    return seq, file_id


# Preprocess the data directory
directory = '/kaggle/temp/data/unlabeled/unlabeled/'
columns = ['sensor_smartwatch', 'sensor_vicon', 'body_part_hand', 'body_part_foot', 'side_right', 'side_left']


test_ids = list(test_metadata['id'])




class ProcessedSequenceDataset(Dataset):
    def __init__(self, data_dir, id_list, max_len, padding_value=0.0):
        """
        Dataset that preprocesses features (scaling and padding) in the initializer.
        Args:
        - id_list: List of IDs for the sequences.
        - max_len: Maximum sequence length for padding.
        """
        self.files = [f'{ID}.csv' for ID in id_list]
        self.id_list = id_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.padding_value = padding_value

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get a single sequence and its file ID.
        """
        filename = self.files[idx]
        seq, file_id = process_sequences(test_metadata, filename, self.id_list, self.data_dir)
        pad_seq = self.pad_to_length(seq, self.max_len, self.padding_value)
        return pad_seq, file_id

    @staticmethod
    def pad_to_length(sequence, target_length, padding_value=0.0):
        seq_length = sequence.size(1)
        if seq_length < target_length:
            padding_needed = target_length - seq_length
            padded_sequence = torch.nn.functional.pad(sequence, (0, padding_needed), value=padding_value)
        else:
            padded_sequence = sequence[:, :target_length]
        return padded_sequence

    @staticmethod
    def collate_fn(batch):
        sequences, file_ids = zip(*batch)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, file_ids
data_dir = '/kaggle/temp/data/unlabeled/unlabeled/'

# Initialize the test dataset and DataLoader
test_dataset = ProcessedSequenceDataset(data_dir=directory, id_list=test_ids, max_len=max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ProcessedSequenceDataset.collate_fn, num_workers=4)







from tqdm import tqdm

# Set the model to evaluation mode
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]

# Start evaluation
all_preds = []
all_ids = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Progress bar added using tqdm
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", leave=True):
        x, ids = batch
        x = x.to(device)

        # Get predictions (probabilities)
        y_hat = torch.nn.functional.softmax(model(x), dim=1)  # Use softmax for probabilities

        # Store predictions and IDs
        all_preds.append(y_hat.cpu())
        all_ids.extend(ids)

# Combine predictions and IDs
all_preds = torch.cat(all_preds).numpy()

# Create the DataFrame for submission
test_results = pd.DataFrame(all_preds, columns=activity_classes)
test_results.insert(0, "sample_id", all_ids)  # Add sample_id as the first column

# Save predictions to a CSV file
submission_path = "/kaggle/working/sample_submission_clean_lstm.csv"
test_results.to_csv(submission_path, index=False)

# Log results to MLflow
with mlflow.start_run():
    mlflow.log_artifact(submission_path)
    print(f"Test predictions saved to {submission_path} and logged to MLflow.")



# Load the CSV file
file_path = "/kaggle/working/sample_submission_clean_lstm.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

output_path = "/kaggle/working/sample_submission_clean_lstm.csv"
df_filled = df.ffill()

# Save the updated DataFrame to a new CSV
df_filled.to_csv(output_path, index=False)

print(f"File saved to {output_path}")


class ProcessedSequenceDatasetMasking(Dataset):
    def __init__(self, data_dir, id_list, max_len, padding_value=0.0, mask_prob = 0.2):
        """
        Dataset that preprocesses features (scaling and padding) in the initializer.
        Args:
        - features: List of DataFrames (one per sequence).
        - labels: List of corresponding labels for the sequences.
        - padding_value: Value to use for padding.
        """
        #convert features to tensors
        self.files = [f'{ID}.csv' for ID in id_list]
        self.padding_value = padding_value
        self.id_list = id_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        """
        Get a single sequence, its mask, and the original sequence for unsupervised training.
        Only masks values from the first 3 columns.
        """
        filename = self.files[idx]
        seq, _ = process_sequences(train, filename, self.id_list, self.data_dir)

        # Create a padded sequence and the padding mask
        pad_seq = pad_to_length(seq, self.max_len, self.padding_value)
        padding_mask = (torch.tensor(pad_seq) == 0.0)  # Mask for padding values

        # Create a random mask only for the first 3 columns (seq_len, 3)
        random_mask = torch.rand(pad_seq.shape[0], 3) < self.mask_prob  # Shape (seq_len, 3)

        # Masking mask (True for masked positions, False for padding)
        masking_mask = random_mask & ~padding_mask[:, :3]  # Apply mask only to the first 3 columns

        # Expand the masking mask to the shape of the entire sequence (pad_seq.shape)
        full_masking_mask = torch.cat([masking_mask, torch.zeros(pad_seq.shape[0], pad_seq.shape[1] - 3, dtype=torch.bool)], dim=1)

        # Apply the mask to the sequence (mask only the first 3 columns)
        masked_seq = pad_seq.clone()  # Make a copy of the padded sequence
        masked_seq[:, :3] = masked_seq[:, :3] * ~full_masking_mask[:, :3]  # Apply mask only to the first 3 columns

        # Return both the masked sequence, original sequence (target), and the masking mask
        return masked_seq, pad_seq, full_masking_mask




data_dir = '/kaggle/temp/data/unlabeled/unlabeled/'
mask_prob = 0.2
train_dataset = ProcessedSequenceDatasetMasking(data_dir, train_ids, max_len, mask_prob = mask_prob)
val_dataset = ProcessedSequenceDatasetMasking(data_dir, val_ids, max_len, mask_prob = mask_prob)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)





class SimpleLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=3, bidirectional=True, learning_rate=0.001):
        super(SimpleLSTM, self).__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, input_dim)  # Output dimension matches input size
        self.learning_rate = learning_rate
        self.print_flag = True

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  # Output sequence, same length as input
        return output

    def training_step(self, batch, batch_idx):
        x, actual, masking = batch  # Get the input (already masked) and the mask
        y_hat = self(x)  # Forward pass on the masked sequence
        # Compute the loss on unmasked positions
        loss = self.compute_loss(y_hat, actual, masking)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_seq, actual, mask = batch  # masked_seq is the input and mask is the true output (without padding)
        # Forward pass
        y_hat = self(masked_seq)
        # Calculate the loss (mean squared error or other loss function suited for sequence generation)
        loss = self.compute_loss(y_hat, actual, mask)
        # Log the loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


    def compute_loss(self, y_hat, original_seq, masking_mask):
        """
        Compute the loss, only on the masked positions, and exclude padding positions.
        """
        # Convert masking mask to float (True becomes 1, False becomes 0)
        masking_mask = masking_mask.float()
        if self.print_flag:
            print_masked_values(y_hat, original_seq, masking_mask)
            self.print_flag = False
        # Use MSE loss for unsupervised prediction task
        loss = nn.MSELoss(reduction="none")(y_hat, original_seq)  # Compute MSE loss for each position

        # Mask the loss to only include the masked positions (where masking_mask is True)
        masked_loss = loss * masking_mask  # Only include loss for masked positions

        res = masked_loss.sum() / masking_mask.sum()

        return res


    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        # Reactivate the flag for debugging the first batch of the next epoch
        self.print_flag = True


import warnings
warnings.filterwarnings("ignore")

# Initialize the model
max_len = train_loader.dataset.max_len  # Defined in `ProcessedSequenceDataset`
input_dim = max_len
hidden_dim = 64
learning_rate = 0.001

model = SimpleLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=2,
    bidirectional=False,
    learning_rate=learning_rate
)
# Setup MLflow
mlflow.set_experiment("Masking_LSTM_Experiment")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="Masking_LSTM_Experiment",
    run_name="LSTM_Run"
)


# Trainer setup
trainer = pl.Trainer(
    max_epochs=7,
    precision=16,
    logger=mlflow_logger,  # Add MLflow logger here
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)
    mlflow.log_param("batch_size", 32)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log metrics to MLflow
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

    # Save the best model to MLflow
    mlflow.pytorch.log_model(model, artifact_path="model")

    print(val_results)


trainer.save_checkpoint("trained_model.ckpt")
checkpoint = torch.load("./trained_model.ckpt")

class ProcessedSequenceDataset(Dataset):
    def __init__(self, data_dir, labels, id_list, max_len, padding_value=0.0):
        """
        Dataset that preprocesses features (scaling and padding) in the initializer.
        Args:
        - features: List of DataFrames (one per sequence).
        - labels: List of corresponding labels for the sequences.
        - padding_value: Value to use for padding.
        """
        #convert features to tensors
        self.files = [f'{ID}.csv' for ID in id_list]
        self.labels = labels
        self.padding_value = padding_value
        self.id_list = id_list
        self.data_dir = data_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        """
        Get a single sequence and its label.
        """
        filename = self.files[idx]
        seq, label =  process_sequences(train, filename, self.id_list, self.data_dir)
        pad_seq = pad_to_length(seq, self.max_len, self.padding_value)
        return pad_seq, label

    def collate_fn(batch):
        sequences, labels = zip(*batch)

        # Pad sequences along the time dimension (dim=0)
        padded_sequences = pad_sequence(sequences, batch_first=True)

        labels = torch.tensor(labels)
        res = (padded_sequences, labels)
        return res



data_dir = '/kaggle/temp/data/unlabeled/unlabeled/'

train_dataset = ProcessedSequenceDataset(data_dir,labels_dict, train_ids,max_len)
val_dataset = ProcessedSequenceDataset(data_dir,labels_dict,val_ids, max_len)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ProcessedSequenceDataset.collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ProcessedSequenceDataset.collate_fn, num_workers=4)

# Setup MLflow
mlflow.set_experiment("PreTrained_LSTM")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="PreTrained_LSTM",
    run_name="LSTM_Run"
)

def freeze_layers(model):
    for name, param in model.named_parameters():
        if "lstm" in name:  # Freeze LSTM layers
            param.requires_grad = False

def unfreeze_layers(model):
    for name, param in model.named_parameters():
        if "lstm" in name:  # Unfreeze LSTM layers
            param.requires_grad = True


# LSTM Model
class FineTunedLSTM(SimpleLSTM):
    def __init__(self, input_dim, hidden_dim, num_classes, freeze_epochs=3, **kwargs):
        super(FineTunedLSTM, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, **kwargs)

        self.freeze_epochs = freeze_epochs  # Number of epochs to freeze layers
        lstm_output_dim = hidden_dim * 2 if self.hparams.bidirectional else hidden_dim
        # self.layer_norm = nn.LayerNorm(lstm_output_dim)  # Layer Normalization
        # self.attention = nn.MultiheadAttention(embed_dim=lstm_output_dim, num_heads=4, batch_first=True)  # Attention layer
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        # norm_out = self.layer_norm(lstm_out)  # Apply layer normalization
        # # Apply self-attention
        # attention_out, _ = self.attention(norm_out, norm_out, norm_out)
        output = self.fc(lstm_out)  # Output sequence, same length as input
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params": self.lstm.parameters(), "lr": 1e-4},  # Lower learning rate for frozen layers
            {"params": self.fc.parameters(), "lr": 1e-3}     # Higher learning rate for new layers
        ])
        return optimizer

    def on_train_epoch_start(self):
        # Freeze layers for the first few epochs
        if self.current_epoch < self.freeze_epochs:
            freeze_layers(self)
            print(f"Epoch {self.current_epoch}: Freezing layers.")
        else:
            unfreeze_layers(self)
            print(f"Epoch {self.current_epoch}: Unfreezing layers.")

# Define input dimensions and other parameters
max_len = train_loader.dataset.max_len  # Defined in `ProcessedSequenceDataset`
input_dim = max_len
hidden_dim = 64
num_classes = len(set(labels_dict.values()))  # Number of unique labels in your dataset
learning_rate = 0.001

# Initialize the model
model = FineTunedLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=2,
    bidirectional=False,
    learning_rate=learning_rate,
    freeze_epochs=5
)

pretrained_dict = checkpoint['state_dict']
model_dict = model.state_dict()

# Filter out the `fc` layer weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}

# Update the current model's weights with the pre-trained ones
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Trainer setup
trainer = pl.Trainer(
    max_epochs=15,
    logger=mlflow_logger,  # Add MLflow logger here
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)
    mlflow.log_param("batch_size", 32)

    # Train the model
    trainer.fit(model, train_loader, val_loader)





from tqdm.notebook import tqdm

# Set the model to evaluation mode
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]

# Start evaluation
all_preds = []
all_ids = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Progress bar added using tqdm
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", leave=True):
        x, ids = batch
        x = x.to(device)

        # Get predictions (probabilities)
        y_hat = torch.nn.functional.softmax(model(x), dim=1)  # Use softmax for probabilities

        # Store predictions and IDs
        all_preds.append(y_hat.cpu())
        all_ids.extend(ids)

# Combine predictions and IDs
all_preds = torch.cat(all_preds).numpy()

# Create the DataFrame for submission
test_results = pd.DataFrame(all_preds, columns=activity_classes)
test_results.insert(0, "sample_id", all_ids)  # Add sample_id as the first column
test_results = test_results.ffill()
# Save predictions to a CSV file
submission_path = "/kaggle/working/sample_submission_fin.csv"
test_results.to_csv(submission_path, index=False)

# Log results to MLflow
with mlflow.start_run():
    mlflow.log_artifact(submission_path)
    print(f"Test predictions saved to {submission_path} and logged to MLflow.")


from pytorch_lightning.callbacks import Callback

class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

# metrics_tracker = MetricsTracker()

# with mlflow.start_run():


#     # Validate the model
#     val_results = trainer.validate(model, val_loader)

#     # Log validation results
#     for metric in val_results:
#         for k, v in metric.items():
#             mlflow.log_metric(k, v)

# # Save metrics to file
# metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")


# Visualizations
# Plot training and validation loss
# metrics = trainer.logged_metrics
# train_losses = metrics["train_loss"]
# val_losses = metrics["val_loss"]
# train_losses = metrics_tracker.train_losses
# validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

# metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
# plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Function to extract predictions from the model
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]
def extract_predictions(model, dataloader, activity_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
# Extract predictions, true labels, and probabilities
predictions, true_labels, probabilities = extract_predictions(model, val_loader, activity_classes)

# High confidence threshold (e.g., >90%)
high_confidence_threshold = 0.9
# Low confidence threshold (e.g., <60%)
low_confidence_threshold = 0.6

# Identify good classifications (correct predictions with high confidence)
good_indices = [
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify bad classifications (incorrect predictions with high confidence)
bad_indices = [
    i for i in range(len(predictions))
    if predictions[i] != true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify uncertain predictions (confidence below threshold)
uncertain_indices = [
    i for i in range(len(predictions))
    if max(probabilities[i]) < low_confidence_threshold
]

# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
    idx)




# Setup MLflow
mlflow.set_experiment("PreTrained_LSTM")

# MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="PreTrained_LSTM",
    run_name="LSTM_Run"
)

def freeze_layers(model):
    for name, param in model.named_parameters():
        if "lstm" in name:  # Freeze LSTM layers
            param.requires_grad = False

def unfreeze_layers(model):
    for name, param in model.named_parameters():
        if "lstm" in name:  # Unfreeze LSTM layers
            param.requires_grad = True


# LSTM Model
class FineTunedLSTM(SimpleLSTM):
    def __init__(self, input_dim, hidden_dim, num_classes, freeze_epochs=3, **kwargs):
        super(FineTunedLSTM, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, **kwargs)

        self.freeze_epochs = freeze_epochs  # Number of epochs to freeze layers
        lstm_output_dim = hidden_dim * 2 if self.hparams.bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)  # Output sequence, same length as input
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params": self.lstm.parameters(), "lr": 1e-4},  # Lower learning rate for frozen layers
            {"params": self.fc.parameters(), "lr": 1e-3}     # Higher learning rate for new layers
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_train_epoch_start(self):
        # Freeze layers for the first few epochs
        if self.current_epoch < self.freeze_epochs:
            freeze_layers(self)
            print(f"Epoch {self.current_epoch}: Freezing layers.")
        else:
            unfreeze_layers(self)
            print(f"Epoch {self.current_epoch}: Unfreezing layers.")

# Define input dimensions and other parameters
max_len = train_loader.dataset.max_len  # Defined in `ProcessedSequenceDataset`
input_dim = max_len
hidden_dim = 64
num_classes = len(set(labels_dict.values()))  # Number of unique labels in your dataset
learning_rate = 0.001

# Initialize the model
model = FineTunedLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=2,
    bidirectional=False,
    learning_rate=learning_rate,
    freeze_epochs=5
)

pretrained_dict = checkpoint['state_dict']
model_dict = model.state_dict()

# Filter out the `fc` layer weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}

# Update the current model's weights with the pre-trained ones
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Trainer setup
trainer = pl.Trainer(
    max_epochs=15,
    logger=mlflow_logger,  # Add MLflow logger here
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=torch.cuda.device_count()
)

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters to MLflow
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", 3)
    mlflow.log_param("bidirectional", False)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_len", max_len)
    mlflow.log_param("batch_size", 32)

    # Train the model
    trainer.fit(model, train_loader, val_loader)





from tqdm.notebook import tqdm

# Set the model to evaluation mode
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]

# Start evaluation
all_preds = []
all_ids = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Progress bar added using tqdm
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", leave=True):
        x, ids = batch
        x = x.to(device)

        # Get predictions (probabilities)
        y_hat = torch.nn.functional.softmax(model(x), dim=1)  # Use softmax for probabilities

        # Store predictions and IDs
        all_preds.append(y_hat.cpu())
        all_ids.extend(ids)

# Combine predictions and IDs
all_preds = torch.cat(all_preds).numpy()

# Create the DataFrame for submission
test_results = pd.DataFrame(all_preds, columns=activity_classes)
test_results = test_results.ffill()
test_results.insert(0, "sample_id", all_ids)  # Add sample_id as the first column

# Save predictions to a CSV file
submission_path = "/kaggle/working/sample_submission_fin.csv"
test_results.to_csv(submission_path, index=False)

# Log results to MLflow
with mlflow.start_run():
    mlflow.log_artifact(submission_path)
    print(f"Test predictions saved to {submission_path} and logged to MLflow.")


from pytorch_lightning.callbacks import Callback

class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the training loss from the logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss and accuracy from the logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss:
            self.val_losses.append(val_loss.item())
        if val_acc:
            self.val_accuracies.append(val_acc.item())

    def save_metrics_to_file(self, file_path):
        # Save metrics to a text file
        with open(file_path, "w") as f:
            f.write("Training and Validation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else "N/A"
                val_loss = self.val_losses[i] if i < len(self.val_losses) else "N/A"
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else "N/A"
                f.write(f"{i+1}, {train_loss}, {val_loss}, {val_acc}\n")
        print(f"Metrics saved to {file_path}")

metrics_tracker = MetricsTracker()

with mlflow.start_run():


    # Validate the model
    val_results = trainer.validate(model, val_loader)

    # Log validation results
    for metric in val_results:
        for k, v in metric.items():
            mlflow.log_metric(k, v)

# # Save metrics to file
metrics_tracker.save_metrics_to_file("/kaggle/working/metrics_log.txt")


# Visualizations
# Plot training and validation loss
# metrics = trainer.logged_metrics
# train_losses = metrics["train_loss"]
# val_losses = metrics["val_loss"]
train_losses = metrics_tracker.train_losses
validation_losses = metrics_tracker.val_losses

# Pad train_losses to match the length of validation_losses
if len(train_losses) < len(validation_losses):
    padding_value = train_losses[-1] if train_losses else 0  # Use last value or 0 if empty
    train_losses += [padding_value] * (len(validation_losses) - len(train_losses))

# metrics_tracker.train_losses = train_losses
# Visualize Training and Validation Loss
plt.figure(figsize=(10, 6))
# plt.plot(metrics_tracker.train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.legend()
plt.grid(True)
plt.show()

# Visualize Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Function to extract predictions from the model
activity_classes = [
    "brushing_teeth", "idle", "preparing_sandwich", "reading_book",
    "stairs_down", "stairs_up", "typing", "using_phone", "using_remote_control",
    "walking_freely", "walking_holding_a_tray", "walking_with_handbag",
    "walking_with_hands_in_pockets", "walking_with_object_underarm",
    "washing_face_and_hands", "washing_mug", "washing_plate", "writing"
]
def extract_predictions(model, dataloader, activity_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
# Extract predictions, true labels, and probabilities
predictions, true_labels, probabilities = extract_predictions(model, val_loader, activity_classes)

# High confidence threshold (e.g., >90%)
high_confidence_threshold = 0.9
# Low confidence threshold (e.g., <60%)
low_confidence_threshold = 0.6

# Identify good classifications (correct predictions with high confidence)
good_indices = [
    i for i in range(len(predictions))
    if predictions[i] == true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify bad classifications (incorrect predictions with high confidence)
bad_indices = [
    i for i in range(len(predictions))
    if predictions[i] != true_labels[i] and max(probabilities[i]) >= high_confidence_threshold
]

# Identify uncertain predictions (confidence below threshold)
uncertain_indices = [
    i for i in range(len(predictions))
    if max(probabilities[i]) < low_confidence_threshold
]

# Function to visualize examples with activity names
def plot_example(sequence, true_label_idx, predicted_label_idx, confidence, title, activity_classes, idx):
    true_label = activity_classes[true_label_idx]
    predicted_label = activity_classes[predicted_label_idx]

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.plot(sequence[:, i], label=f"{axis}-axis")
    plt.title(
        f"{title}\nID: {idx}, True Activity: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize good classifications
print("Good Classifications:")
for idx in good_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Good Classification",
        activity_classes,
        idx
    )

# Visualize bad classifications
print("Bad Classifications:")
for idx in bad_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Bad Classification",
        activity_classes,
        idx
    )

# Visualize uncertain predictions
print("Uncertain Predictions:")
for idx in uncertain_indices[:3]:  # Show up to 3 examples
    seq, _ = val_loader.dataset[idx]
    plot_example(
        seq.numpy().T,
        true_labels[idx],
        predictions[idx],
        max(probabilities[idx]),
        "Uncertain Prediction",
        activity_classes,
    idx)

from nbconvert import HTMLExporter
import nbformat

# Load the notebook
with open("deep_learning_workshop_1_213138787_213479686_(3) (1).ipynb", "r", encoding="utf-8") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to HTML
html_exporter = HTMLExporter()
html_content, _ = html_exporter.from_notebook_node(notebook_content)

# Save HTML to file
with open("deep_learning_workshop_1_213138787_213479686_(3) (1).html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Notebook converted to HTML successfully!")
