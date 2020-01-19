import SMSSpamSupport
from EvaluationsStub import Accuracy
import EvaluationsStub
import matplotlib.pyplot as plt 

def PlotTrainingSetLoss(x, y, xLabel, yLabel , title , fileName):
    # plotting the points
    plt.figure(0)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.plot(x, y) 
    plt.show(block=False)
    plt.savefig(fileName)

def PlotValidationSetLoss(yfeature, ytrainingsetLoss, yAccuracy, x, title , fileName):
    # plotting the points
    plt.figure(1)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title(title)
    plt.plot(x, yfeature, label="weights[2](Second feature weight excluding w0)") 
    plt.plot(x, ytrainingsetLoss, label="Validationset Loss") 
    plt.plot(x, yAccuracy, label="Accuracy") 
    plt.legend() 
    plt.show(block=False)
    plt.savefig(fileName)


### UPDATE this path for your environment
kDataPath = "SMSSpamCollection"

(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xValidateRaw, yValidateRaw, xTestRaw, yTestRaw) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %f percent spam." % (len(yTrainRaw), sum(yTrainRaw) / len(yTrainRaw)))
print("Validate is %d samples, %f percent spam." % (len(yValidateRaw), sum(yValidateRaw) / len(yValidateRaw)))
print("Test is %d samples %f percent spam." % (len(yTestRaw), sum(yTestRaw) / len(yTestRaw)))

(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw, xTestRaw, False)
yTrain = yTrainRaw
yValidate = yValidateRaw
yTest = yTestRaw

############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)

print("### 'Most Common' model")

EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)

print("### Heuristic model")

EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted)

############################

(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw, xTestRaw, True)
yTrain = yTrainRaw
yValidate = yValidateRaw
yTest = yTestRaw

import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

print("Logistic regression model")
trainingsetloss = []
trainingsetiterationcount = []

validationsetLossIterationCount = []
featureWeightSet = []
validationsetLossSet = []
validationsetAccuracySet = []

for i in range(50000):
    model.fit(xTrain, yTrain, iterations=i, step=0.01)
    yValidatePredicted = model.predict(xValidate)
    featureWeight = model.featureWeights[2]
    validationSetLoss = model.loss(xValidate,yValidate)
    validationsetAccuracy = EvaluationsStub.Accuracy(yValidate, yValidatePredicted)
    print("%d, %f, %f, %f" % (i, featureWeight, validationSetLoss, validationsetAccuracy))

    if i % 1000 == 0:
        trainingsetiterationcount.append(i)
        trainingsetloss.append(model.loss(xTrain,yTrain))

    if i % 1000 == 0:
        validationsetLossIterationCount.append(i)
        featureWeightSet.append(featureWeight)
        validationsetLossSet.append(validationSetLoss)
        validationsetAccuracySet.append(validationsetAccuracy)

PlotTrainingSetLoss(trainingsetiterationcount, trainingsetloss,"Iteration count","Training set loss", "training set loss vs iteration","trainingloss.png")
PlotValidationSetLoss(featureWeightSet,validationsetLossSet,validationsetAccuracySet,validationsetLossIterationCount,"Validation Set and Stats","validationloss.png")
EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted)

# Don't use the test data yet...save that for after we do some more serious feature engineering.