import tqdm as tq
import torch
import torch.nn as nn
from torch import cuda
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data_loader import NiftyDatasetLoader
from utils import calculate_TIs, create_data
import pickle
from models.cnn import TA_CNN
from models.lstm import TA_LSTM
from models.transformer import TA_Transformer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
test_year = "2017"
days = 10
batch_size = 8
device = 'cuda' if cuda.is_available() else 'cpu'
data_path = "../dataset/"  # Add the path to dataset
saved_path = "../transformer/finbert/" # Add the path where model is to be saved
classifier = "transformer" # Type of classifier to be used
file_name = "NIFTY_50.csv" # File name containing prices (OHLC) for a index
learning_rate = 1e-2 # Learning rate
epochs = 20


nifty_df = pd.read_csv(data_path + file_name)
nifty_df = calculate_TIs(nifty_df)
scaler = StandardScaler()
train_df = create_data(nifty_df, train_years)
train_df.iloc[::,1:-1] = scaler.fit_transform(train_df.iloc[::,1:-1].values)

nifty_train_data = NiftyDatasetLoader(train_df, days)
nifty_train_data_loader = DataLoader(nifty_train_data, batch_size=batch_size, shuffle=False)

if classifier == "lstm":
    nifty_prediction_net = TA_LSTM()
elif classifier == "cnn":
    nifty_prediction_net = TA_CNN()
elif classifier == "transformer":
    nifty_prediction_net = TA_Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = Adam(nifty_prediction_net.parameters(), lr=learning_rate)

nifty_prediction_net = nifty_prediction_net.to(device)
nifty_prediction_net.train()
for epoch in range(epochs):
    print("Epoch no {:} of {:}".format(epoch+1, epochs))
    optimizer.zero_grad()
    train_loss = 0
    for i, train_data in enumerate(tq.tqdm(nifty_train_data_loader)):
        prev_nifty_values = train_data[0].to(device)
        label = train_data[1].to(device)
        label = label.squeeze(1)
        logits = nifty_prediction_net(prev_nifty_values)
        logits = logits.squeeze(1)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_training_loss = train_loss/len(nifty_train_data_loader)
    print("Loss at Epoch no {:} of {:} is {:}".format(epoch+1, epochs, avg_training_loss))


torch.save(nifty_prediction_net.state_dict(), saved_path + "{:}_{:}.pt".format(classifier, test_year))
fp = open(saved_path + "scaler_{:}.pkl".format(test_year), "wb")
pickle.dump(scaler, fp)
