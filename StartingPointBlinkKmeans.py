## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL') via 'Install Pillow' or 'pip install Pillow' depending on your environment

import BlinkSupport

kDataPath = "..\\..\\..\\Datasets\\Blink"

(xRaw, yRaw) = BlinkSupport.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xValidateRaw, yValidateRaw, xTestRaw, yTestRaw) = BlinkSupport.TrainValidateTestSplit(xRaw, yRaw, percentValidate = .10, percentTest = .15)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xValidate, xTest) = BlinkSupport.Featurize(xTrainRaw, xValidateRaw, xTestRaw, verbose=True, includeGradients=True, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yValidate = yValidateRaw
yTest = yTestRaw


## REMEMBER to update the references to validations and modeling to use your implementations
import Evaluations
import ErrorBounds

######
import MostCommonModel
model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)
print("Most Common Accuracy:", Evaluations.Accuracy(yValidate, yValidatePredicted), ErrorBounds.Get95LowerAndUpperBounds(Evaluations.Accuracy(yValidate, yValidatePredicted), len(yValidate)))

######
import DecisionTreeModel
model = DecisionTreeModel.DecisionTree()
model.fit(xTrain, yTrain, minToSplit=10)
yValidatePredictedPredicted = model.predict(xValidate)
print("Decision Tree Accuracy:", Evaluations.Accuracy(yValidate, yValidatePredicted), ErrorBounds.Get95LowerAndUpperBounds(Evaluations.Accuracy(yValidate, yValidatePredicted), len(yValidate)))


##### sample image debugging output

#import PIL
#from PIL import Image

#i = Image.open(xTrainRaw[1])
#i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")
#print(i.format, i.size)

