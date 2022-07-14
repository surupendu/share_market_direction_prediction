import tqdm as tq
import pandas as pd
from torch.utils.data import DataLoader
from torch import cuda
from data_loader import NiftyDatasetLoader
from ta_fa_model import TA_Nifty_Prediction
from utils import create_data, calculate_cross_corr
from technical_indicators import get_indicators
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

data_path = "../dataset/"
saved_path = "../ta_fa_predictions/"
file_name = "NIFTY_50.csv"
max_length = 50
days = 7
batch_size = 1
train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
test_year = '2017'
start_time = "09:00"
end_time = "16:00"
tokenizer = "finbert"
text_representation = "finbert"

nifty_df = pd.read_csv(data_path + file_name)
news_df = pd.read_csv(data_path + "all_close_news.csv", lineterminator="\n")
nifty_df = get_indicators(nifty_df)

# fp = open(saved_path + "scaler_{:}.pkl".format(test_year), 'rb')
# scaler = pickle.load(fp)

train_df = create_data(nifty_df, train_years) 
test_df = create_data(nifty_df, [test_year])
# print(test_df.head())

scaler = StandardScaler()
train_df.iloc[::,1:-1] = scaler.fit_transform(train_df.iloc[::,1:-1].values)
test_df.iloc[::,1:-1] = scaler.transform(test_df.iloc[::,1:-1].values)

nifty_test_data = NiftyDatasetLoader(test_df, news_df, max_length, days, tokenizer)
nifty_test_data_loader = DataLoader(nifty_test_data, batch_size=batch_size, shuffle=False)

device = 'cuda' if cuda.is_available() else 'cpu'

nifty_prediction_net = TA_Nifty_Prediction(text_representation)
nifty_prediction_net.load_state_dict(torch.load(saved_path + "ta_fa_model_{:}.pt".format(test_year)))
# nifty_prediction_net.load_state_dict(torch.load("/home/irlab/Documents/Share/Surupendu/share_market_prediction/ta_fa_predictions/ta_fa_model_2017.pt"))

nifty_prediction_net.to(device)

nifty_prediction_net.eval()
actual_labels = []
pred_labels = []
with torch.no_grad():
    for test_data in tq.tqdm(nifty_test_data_loader):
        input_ids = test_data[0].to(device)
        attention_mask = test_data[1].to(device)
        prev_nifty_values = test_data[2].to(device)
        label = test_data[3]
        logits = nifty_prediction_net(prev_nifty_values, input_ids, attention_mask)
        probs = F.softmax(logits)
        pred_label = torch.argmax(probs).cpu().item()
        pred_labels.append(pred_label)
        actual_label = label.item()
        actual_labels.append(actual_label)

print("Evaluation of model for year {:}".format(test_year))
accuracy = accuracy_score(actual_labels, pred_labels)
print("Accuracy: {:}".format(accuracy))
lag, max_cross_corr, cross_corr, idxs = calculate_cross_corr(nifty_df, [test_year], actual_labels, pred_labels)
print("Lag: {:} and cross correlation: {:}".format(lag, max_cross_corr))
roc_auc = roc_auc_score(actual_labels, pred_labels)
print("ROC-AUC score: {:}".format(roc_auc))
