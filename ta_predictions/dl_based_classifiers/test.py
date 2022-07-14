import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import NiftyDatasetLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import calculate_cross_corr
from utils import calculate_TIs, create_data
from torch import cuda
from models.cnn import TA_CNN
from models.lstm import TA_LSTM
from models.dnn import TA_DNN
from models.transformer import TA_Transformer
import pandas as pd
import pickle
import tqdm as tq
import warnings

warnings.filterwarnings("ignore")

test_year = "2017"
days = 10
batch_size = 8
device = 'cuda' if cuda.is_available() else 'cpu'
data_path = "/media/sda1/Share/Surupendu/share_market_prediction/dataset/"
saved_path = "/media/sda1/Share/Surupendu/share_market_prediction/saved_wgts/nifty_50/prices/transformer/"
classifier = "transformer"
file_name = "NIFTY_50.csv"

nifty_df = pd.read_csv(data_path + file_name)
nifty_df = calculate_TIs(nifty_df)

fp = open(saved_path + "scaler_{:}.pkl".format(test_year), "rb")
scaler = pickle.load(fp)
test_df = create_data(nifty_df, [test_year])
test_df.iloc[::,1:-1] = scaler.transform(test_df.iloc[::,1:-1].values)

nifty_test_data = NiftyDatasetLoader(test_df, days)
nifty_test_data_loader = DataLoader(nifty_test_data, batch_size=1, shuffle=False)

if classifier == "lstm":
    nifty_prediction_net = TA_LSTM()
elif classifier == "cnn":
    nifty_prediction_net = TA_CNN()
elif classifier == "transformer":
    nifty_prediction_net = TA_Transformer()
elif classifier == "dnn":
    nifty_prediction_net = TA_DNN()

nifty_prediction_net.load_state_dict(torch.load("{:}_{:}.pt".format(saved_path + classifier, test_year)))
nifty_prediction_net.to(device)

actual_labels = []
pred_labels = []

with torch.no_grad():
    for test_data in tq.tqdm(nifty_test_data_loader):
        prev_nifty_values = test_data[0].to(device)
        label = test_data[1].to(device)
        label = label.squeeze(1)
        logits = nifty_prediction_net(prev_nifty_values)
        probs = F.softmax(logits)
        pred_label = torch.argmax(probs).cpu().item()
        pred_labels.append(pred_label)
        actual_label = label.item()
        actual_labels.append(actual_label)

print("Evaluation of model for year {:}".format(test_year))
accuracy = accuracy_score(actual_labels, pred_labels)
print("Accuracy: {:}".format(accuracy))
lag, max_cross_corr, cross_corr, idxs = calculate_cross_corr(test_df, actual_labels, pred_labels)
print("Lag: {:} and cross correlation: {:}".format(lag, max_cross_corr))
roc_auc = roc_auc_score(actual_labels, pred_labels)
print("ROC AUC score: {:}".format(roc_auc))