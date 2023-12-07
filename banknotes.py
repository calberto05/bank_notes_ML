"""
AUTHORS: Oliver Burguete López (A01026488),
         Carlos Alfonso Alberto Salazar (A01026175) &
         Alejandro Méndez Godoy (A01783325)
"""

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


PERCENT_TO_KEEP_FOR_TEST = 0.3


class BanknoteClassifier:
    def __init__(self):
        self.training_data = None
        self.testing_data = None
        self.training_labels = None
        self.testing_labels = None
        self.modeloKNC = None
        self.modeloSVC = None
        self.modeloP = None
        self.modeloTree = None

    def read_data(self, filename: str = "./banknotes.csv"):
        # Read data in from file
        with open(filename) as f:  # this should be the path to the banknotes.csv
            reader = csv.reader(f)
            next(reader)

            data = []
            for row in reader:
                data.append({
                    "evidence": [float(cell) for cell in row[:4]],
                    "label": "Authentic" if row[4] == "0" else "Counterfeit"
                })

        # Separate data into training and testing groups
        evidence = [row["evidence"] for row in data]
        labels = [row["label"] for row in data]

        X_training, X_testing, y_training, y_testing = train_test_split(
            evidence, labels, test_size=PERCENT_TO_KEEP_FOR_TEST,
        )
        self.training_data = X_training
        self.testing_data = X_testing
        self.training_labels = y_training
        self.testing_labels = y_testing

    def train(self):
        """This should be implemented for SVC, KNeighborsClassifier, Perceptron, and one other of your choosing.
        Feel free to change the signature of this method however you see fit, or to add new methods.
        """
        # KNeighborsClassifier
        modelKNC = KNeighborsClassifier()
        modelKNC.fit(self.training_data, self.training_labels)
        self.modeloKNC = modelKNC

        # SVC
        modelSVC = SVC()
        modelSVC.fit(self.training_data, self.training_labels)
        self.modeloSVC = modelSVC

        # Perceptron
        modelP = Perceptron()
        modelP.fit(self.training_data, self.training_labels)
        self.modeloP = modelP

        # Decision Tree
        modelTree = DecisionTreeClassifier()
        modelTree.fit(self.training_data, self.training_labels)
        self.modeloTree = modelTree

    def predict_on_test_set(self):
        # KNC
        predictions_KNC = self.modeloKNC.predict(self.testing_data)
        accuracy_KNC = accuracy_score(self.testing_labels, predictions_KNC)
        print('Results for model KNeighborsClassifier')
        print('Accuracy: {}'.format(accuracy_KNC))
        validation = pd.DataFrame({'Actual': self.testing_labels, 'Predicción': predictions_KNC})
        difference = validation[validation['Actual'] != validation['Predicción']]
        print(f"Correct:, {(len(validation)-len(difference))}\nIncorrect: {(len(difference))}\n{difference}", end="\n\n")

        # SVC
        predictions_SVC = self.modeloSVC.predict(self.testing_data)
        accuracy_SVC = accuracy_score(self.testing_labels, predictions_SVC)
        print('Results for model SVC')
        print('Accuracy: {}'.format(accuracy_SVC))
        validationSVC = pd.DataFrame({'Actual': self.testing_labels, 'Predicción': predictions_SVC})
        differenceSVC = validationSVC[validationSVC['Actual'] != validationSVC['Predicción']]
        print(f"Correct:, {(len(validationSVC)-len(differenceSVC))}\nIncorrect: {(len(differenceSVC))}\n{differenceSVC}", end="\n\n")

        # Perceptron
        predictions_Per = self.modeloP.predict(self.testing_data)
        accuracy_Per = accuracy_score(self.testing_labels, predictions_Per)
        print('Results for model Perceptron')
        print('Accuracy: {}'.format(accuracy_Per))
        validationPer = pd.DataFrame({'Actual': self.testing_labels, 'Predicción': predictions_Per})
        differencePer = validationPer[validationPer['Actual'] != validationPer['Predicción']]
        print(f"Correct:, {(len(validationPer) - len(differencePer))}\nIncorrect: {(len(differencePer))}\n{differencePer}", end="\n\n")

        # Decision Tree
        predictions_Tree = self.modeloTree.predict(self.testing_data)
        accuracy_Tree = accuracy_score(self.testing_labels, predictions_Tree)
        print('Results for model Decision Tree')
        print('Accuracy: {}'.format(accuracy_Tree))
        validationTree = pd.DataFrame({'Actual': self.testing_labels, 'Predicción': predictions_Tree})
        differenceTree = validationTree[validationTree['Actual'] != validationTree['Predicción']]
        print(f"Correct:, {(len(validationTree) - len(differenceTree))}\nIncorrect: {(len(differenceTree))}\n{differenceTree}", end="\n\n")


if __name__ == '__main__':
    classifier = BanknoteClassifier()
    classifier.read_data()
    classifier.train()
    classifier.predict_on_test_set()
