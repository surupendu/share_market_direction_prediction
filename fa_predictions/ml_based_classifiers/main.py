import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from classifiers import NiftyStatClassifier
from utils import create_labelled_data, create_news_data, calculate_cross_corr

train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
test_years = []
all_test_years = ['2017', '2018', '2019', '2020', '2021']

path = "../dataset/" # Path where dataset is present
classifier_name = "svm" # Name of classifier
file_name = "NIFTY_Next_50.csv" # CSV file containing the prices (OHLC)

for test_year in all_test_years:
  test_years = [test_year]
  nifty_df = pd.read_csv(path + file_name)
  news_df = pd.read_csv(path + "all_close_news.csv", lineterminator="\n")

  start_time = "09:00"
  end_time = "16:00"

  nifty_df = create_labelled_data(nifty_df)
  train_docs, train_labels, test_docs, test_labels = create_news_data(nifty_df, news_df, train_years, test_years, start_time, end_time, close=True)

  nifty_stat_classifier = NiftyStatClassifier(classifier=classifier_name)
  nifty_stat_classifier.fit(train_docs, train_labels)
  pred_labels = nifty_stat_classifier.predict(test_docs)

  print("Evaluation of year: {:}".format(test_years[0]))
  accuracy = accuracy_score(test_labels, pred_labels)
  print("Accuracy {:}".format(accuracy))
  lag, max_cross_corr, cross_corr, idxs = calculate_cross_corr(nifty_df, test_years, test_labels, pred_labels)
  print("Lag: {:} and cross correlation: {:}".format(lag, max_cross_corr))
  roc_auc = roc_auc_score(test_labels, pred_labels)
  print("ROC-AUC score {:}".format(roc_auc))
  train_years += test_years