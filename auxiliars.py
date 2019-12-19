from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import pandas as pd

def confusionMatrix(real, predicted, classes):
	"""
	"""

	confusionMat = pd.DataFrame(confusion_matrix(real, predicted),
								index = classes,
								columns = classes)
	confusionMat.index.name = "Real"
	confusionMat.columns.name = "Predicted"

	return confusionMat

def loocv(X, y, model, classes):
    """
    """

    loo = LeaveOneOut()

    predictions = []
    for train_index, test_index in loo.split(X):
        X_training, X_test = X.iloc[train_index], X.iloc[test_index]
        y_training, _ = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_training, y_training)

        predictions.append(model.predict(X_test)[0])
    return confusionMatrix(y, predictions, classes), 1-accuracy_score(y, predictions)
