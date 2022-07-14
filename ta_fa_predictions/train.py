from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import NiftyDatasetLoader
import tqdm as tq
from torch import cuda
from torch.optim import Adam
import torch.nn as nn
from technical_indicators import get_indicators
from ta_fa_model import TA_Nifty_Prediction
from utils import create_data
import torch
import warnings

warnings.filterwarnings("ignore")

data_path = "../dataset/" # Set path to dataset
saved_path = "../parallel_cnn/finbert/" # Set path where model has to be saved
file_name = "NIFTY_Next_50.csv" # Set filename of csv file where prices are stored (OHLC)
max_length = 50 # Set max length of news articles
days = 7 # Set number of days 
batch_size = 1 # Do not change batch size
learning_rate = 1e-5 # Set 1e-5 for NIFTY 50, 2e-5 for NIFTY Bank, 3e-5 for NIFTY Next 50
epochs = 7 # Set number of epochs according to your requirement
accumulation_steps = 8 # Set batch size here
train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
test_year = '2017'
start_time = "09:00"
end_time = "16:00"
tokenizer = "finbert" # Set tokenizer (bert or finbert)
text_representation = "finbert" # Set text representation (bert or finbert)

nifty_df = pd.read_csv(data_path + file_name)
news_df = pd.read_csv(data_path + "all_close_news.csv", lineterminator="\n")

scaler = StandardScaler()
nifty_df = get_indicators(nifty_df)

train_df = create_data(nifty_df, train_years) 
train_df.iloc[::,1:-1] = scaler.fit_transform(train_df.iloc[::,1:-1].values)

nifty_train_data = NiftyDatasetLoader(train_df, news_df, max_length, days, tokenizer)
nifty_train_data_loader = DataLoader(nifty_train_data, batch_size=batch_size, shuffle=False)

nifty_prediction_net = TA_Nifty_Prediction(text_representation)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(nifty_prediction_net.parameters(), lr=learning_rate)

device = 'cuda' if cuda.is_available() else 'cpu'

nifty_prediction_net = nifty_prediction_net.to(device)

for epoch in range(epochs):
  print("Epoch no {:} of {:}".format(epoch+1, epochs))
  optimizer.zero_grad()
  train_loss = 0
  for i, train_data in enumerate(tq.tqdm(nifty_train_data_loader)):
    input_ids = train_data[0].to(device)
    attention_mask = train_data[1].to(device)
    prev_nifty_values = train_data[2].to(device)
    label = train_data[3].to(device)
    label = label.squeeze(1)
    logits = nifty_prediction_net(prev_nifty_values, input_ids, attention_mask)
    loss = criterion(logits, label)
    loss.backward()
    train_loss += loss.item()
    if (i+1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
  avg_training_loss = train_loss/len(nifty_train_data_loader)
  print("Loss at Epoch no {:} of {:} is {:}".format(epoch+1, epochs, avg_training_loss))

torch.save(nifty_prediction_net.state_dict(), saved_path + "ta_fa_model_{:}.pt".format(test_year))