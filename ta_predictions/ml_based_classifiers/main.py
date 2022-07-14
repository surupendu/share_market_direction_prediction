import pandas as pd
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from utils import calculate_TIs, calculate_cross_corr, create_data
from classifiers import NiftyStatTAClassifier

warnings.filterwarnings("ignore")

train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
all_test_years = ['2017', '2018', '2019', '2020', '2021']
days = 10
classifier = "perceptron"
data_path = "/media/sda1/Share/Surupendu/share_market_prediction/dataset/"

for test_year in all_test_years:
  nifty_df = pd.read_csv(data_path + "NIFTY_Next_50.csv")
  nifty_df = calculate_TIs(nifty_df)
  scaler = StandardScaler()
  train_df = create_data(nifty_df, train_years)
  test_df = create_data(nifty_df, [test_year])
  train_df.iloc[::,1:-1] = scaler.fit_transform(train_df.iloc[::,1:-1].values)
  test_df.iloc[::,1:-1] = scaler.transform(test_df.iloc[::,1:-1].values)
  X_train, X_labels = [], []
  Y_test, Y_labels = [], []
  for idx in range(0, len(train_df)-days):
    features = train_df.iloc[idx:idx+days]
    features = np.array(features.iloc[::,1:-1].values)
    features = features.flatten()
    X_train.append(features)
    label = train_df.iloc[idx+days]["Label"]
    X_labels.append(label)
  X_train = np.array(X_train)
  X_labels = np.array(X_labels)
  for idx in range(0, len(test_df)-days):
    features = test_df.iloc[idx:idx+days]
    features = np.array(features.iloc[::,1:-1].values)
    features = features.flatten()
    Y_test.append(features)
    label = test_df.iloc[idx+days]["Label"]
    Y_labels.append(label)
  Y_test = np.array(Y_test)
  Y_labels = np.array(Y_labels)
  ta_classifier = NiftyStatTAClassifier(classifier)
  ta_classifier.fit(X_train, X_labels)
  pred_labels = ta_classifier.predict(Y_test)
  accuracy = accuracy_score(Y_labels, pred_labels)

  print("Evaluation of model for year {:}".format(test_year))
  print("Accuracy {:}".format(accuracy))
  lag, max_cross_corr, cross_corr, idxs = calculate_cross_corr(test_df, Y_labels, pred_labels)
  print("Lag: {:} and cross correlation: {:}".format(lag, max_cross_corr))
  roc_auc = roc_auc_score(Y_labels, pred_labels)
  print("ROC AUC score: {:}".format(roc_auc))