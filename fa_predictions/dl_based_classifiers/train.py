from torch.optim import Adam
from utils import create_data, create_labelled_data

from models.cnn import Predict_CNN
from models.lstm import Predict_LSTM
from models.transformer_model import Predict_Transformer

import pandas as pd
from data_loader import NiftyDatasetLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import cuda
import tqdm as tq
import torch

max_length = 50 # Max length of title of news article
accumulation_steps = 8 # For changing the batch size use this variable
batch_size = 1 # Do not change this
data_path = "../dataset/"  # Add the path to dataset
saved_path = "../transformer/finbert/" # Add the path where model is to be saved
classifier_name = "transformer" # Type of classifier to be used
tokenizer = "finbert" # Type of tokenizer (finbert or bert)
text_representation = "finbert" # Type of representation (finbert or bert)
learning_rate = 1e-5 # Learning rate of NIFTY 50 is 1e-5, NIFTY Next 50 is 3e-5 and NIFTY Bank is 2e-5
file_name = "NIFTY_50.csv" # File name containing prices (OHLC) for a index
epochs = 4 # Change number of epochs according to your requirement

train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
test_year = "2017"

test_years = [test_year]
nifty_df = pd.read_csv(data_path + file_name)
news_df = pd.read_csv(data_path + "all_close_news.csv", lineterminator="\n")
start_time = "09:00"
end_time = "16:00"

device = 'cuda' if cuda.is_available() else 'cpu'

if classifier_name == "transformer":
    print("Prediction Model Transformer")
    predict_nn = Predict_Transformer()
if classifier_name == "cnn":
    print("Prediction Model CNN")
    predict_nn = Predict_CNN()
if classifier_name == "lstm":
    print("Prediction Model LSTM")
    predict_nn = Predict_LSTM()

optimizer = Adam(predict_nn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
predict_nn = predict_nn.to(device)

nifty_df = create_labelled_data(nifty_df)
train_df = create_data(nifty_df, train_years)
nifty_train_data = NiftyDatasetLoader(train_df, news_df, max_length, start_time, end_time, classifier_name, tokenizer=tokenizer)
nifty_train_data_loader = DataLoader(nifty_train_data, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    print("Epoch no {:} of {:}".format(epoch+1, epochs))
    optimizer.zero_grad()
    train_loss = 0
    for i, train_data in enumerate(tq.tqdm(nifty_train_data_loader)):
      input_ids = train_data[0].to(device)
      attention_mask = train_data[1].to(device)
      label = train_data[2].to(device)
      logits = predict_nn(input_ids, attention_mask)
      loss = criterion(logits, label.squeeze(0))
      loss = loss/accumulation_steps
      train_loss += loss.item()
      loss.backward()
      if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    avg_training_loss = train_loss/len(nifty_train_data_loader)
    print("Loss at Epoch no {:} of {:} is {:}".format(epoch+1, epochs, avg_training_loss))
  
torch.save(predict_nn.state_dict(), saved_path + "{:}_{:}.pt".format(classifier_name, test_year))
