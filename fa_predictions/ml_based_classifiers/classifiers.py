from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer

class NiftyStatClassifier:
  def __init__(self, classifier):
    self.tf_idf = TfidfVectorizer()
    if classifier == "svm":
        print("Prediction Model SVM")
        self.classifier = LinearSVC()
    if classifier == "random_forest":
        print("Prediction Model Random Forest")
        self.classifier = RandomForestClassifier()
    if classifier == "perceptron":
        print("Prediction Model Perceptron")
        self.classifier = Perceptron(max_iter=40)

  def fit(self, train_docs, labels):
    X = self.tf_idf.fit_transform(train_docs)
    tf_idf_docs = self.tf_idf.transform(train_docs)
    self.classifier = self.classifier.fit(tf_idf_docs, labels)
  
  def predict(self, test_docs):
    test_tf_idf_docs = self.tf_idf.transform(test_docs)
    pred_labels = self.classifier.predict(test_tf_idf_docs)
    return pred_labels