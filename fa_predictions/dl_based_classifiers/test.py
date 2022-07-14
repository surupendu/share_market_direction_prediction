from data_loader import NiftyDatasetLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import create_data, create_labelled_data, calculate_cross_corr
import pandas as pd
import torch
from torch import cuda
import tqdm as tq
from models.cnn import Predict_CNN
from models.lstm import Predict_LSTM
from models.transformer_model import Predict_Transformer
from models.dnn import Predict_DNN

test_year = "2017"
max_length = 50
batch_size = 1
data_path = "../dataset/"
saved_path = "../saved_wgts/nifty_next_50/dnn/finbert/"
classifier_name = "dnn"
tokenizer = "finbert"
text_representation = "finbert"
file_name = "NIFTY_Next_50.csv"

test_years = [test_year]
nifty_df = pd.read_csv(data_path + file_name)
news_df = pd.read_csv(data_path + "all_close_news.csv", lineterminator="\n")
start_time = "09:00"
end_time = "16:00"

nifty_df = create_labelled_data(nifty_df)
test_df = create_data(nifty_df, test_years)
nifty_test_data = NiftyDatasetLoader(test_df, news_df, max_length, start_time, end_time, classifier_name, tokenizer=tokenizer)
nifty_test_data_loader = DataLoader(nifty_test_data, batch_size=1, shuffle=False)

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
if classifier_name == "dnn":
    print("Prediction Model DNN")
    predict_nn = Predict_DNN(text_representation)

predict_nn.load_state_dict(torch.load("{:}_{:}.pt".format(saved_path + classifier_name, test_year)))
predict_nn.to(device)
predict_nn.eval()

actual_labels = []
pred_labels = []

with torch.no_grad():
    for test_data in tq.tqdm(nifty_test_data_loader):
        input_ids = test_data[0].to(device)
        attention_mask = test_data[1].to(device)
        label = test_data[2].to(device)
        label = label.squeeze(1)
        label = label.to(device)
        logits = predict_nn(input_ids, attention_mask)
        probs = F.softmax(logits)
        pred_label = torch.argmax(probs).cpu().item()
        pred_labels.append(pred_label)
        actual_label = label.cpu().item()
        actual_labels.append(actual_label)


print("Evaluation of model for year {:}".format(test_year))
accuracy = accuracy_score(actual_labels, pred_labels)
print("Accuracy: {:}".format(accuracy))
lag, max_cross_corr, cross_corr, idxs = calculate_cross_corr(nifty_df, test_years, actual_labels, pred_labels)
print("Lag: {:} and cross correlation: {:}".format(lag, max_cross_corr))
roc_auc = roc_auc_score(actual_labels, pred_labels)
print("ROC AUC score: {:}".format(roc_auc))


