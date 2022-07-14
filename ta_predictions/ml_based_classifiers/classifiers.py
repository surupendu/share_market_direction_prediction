from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

class NiftyStatTAClassifier:
  def __init__(self, classifier):
    if classifier == "svm":
      self.classifier = LinearSVC()
    if classifier == "random_forest":
      self.classifier = RandomForestClassifier()
    if classifier == "perceptron":
      self.classifier = Perceptron(max_iter=40)

  def fit(self, price_features, labels):
    self.classifier = self.classifier.fit(price_features, labels)
  
  def predict(self, price_features):
    pred_labels = self.classifier.predict(price_features)
    return pred_labels
