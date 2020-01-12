####
#	Autores: Ferran Velasco y Joaquin Gomez
####

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
