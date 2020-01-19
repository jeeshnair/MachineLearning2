import SMSSpamSupport
import EvaluationsStub
from joblib import Parallel, delayed

def PerformGradientDescentIteration(model , xTrain,yTrain,xValidate,yValidate,iterationTitle):
    for i in range(50000):
        model.fit(xTrain, yTrain, iterations=i, step=0.01)
        yValidatePredicted = model.predict(xValidate)
        featureWeight = model.featureWeights[2]
        validationSetLoss = model.loss(xValidate,yValidate)
        validationsetAccuracy = EvaluationsStub.Accuracy(yValidate, yValidatePredicted)

    print(iterationTitle)
    print(EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted))

### UPDATE this path for your environment
kDataPath = "SMSSpamCollection"
(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)
(xTrainRaw, yTrainRaw, xValidateRaw, yValidateRaw, xTestRaw, yTestRaw) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %f percent spam." % (len(yTrainRaw), sum(yTrainRaw) / len(yTrainRaw)))
print("Validate is %d samples, %f percent spam." % (len(yValidateRaw), sum(yValidateRaw) / len(yValidateRaw)))
print("Test is %d samples %f percent spam." % (len(yTestRaw), sum(yTestRaw) / len(yTestRaw)))

(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw, xTestRaw, True)
yTrain = yTrainRaw
yValidate = yValidateRaw
yTest = yTestRaw

gradientDescentVariationParams = []
###################################################################################################
##########Feature Selection frequency
##########based########################################################
###################################################################################################
wordCounts = SMSSpamSupport.TokenizeFeatures(xTrainRaw)
words = SMSSpamSupport.GetFeaturesByFrequency(wordCounts, top = 10)

gradientDescentParam={}
gradientDescentParam["words"] = words
gradientDescentParam["featureCount"]= 10
gradientDescentParam["featureText"] = "High Frequency"
gradientDescentVariationParams.append(gradientDescentParam)

print(words)

###################################################################################################
##########Feature Selection MI
##########based###############################################################
###################################################################################################
words = SMSSpamSupport.GetFeaturesByMutualInformation(xTrainRaw,yTrain,wordCounts,top=10)

gradientDescentParam={}
gradientDescentParam["words"] = words
gradientDescentParam["featureCount"]= 10
gradientDescentParam["featureText"] = "Mutual Index"
gradientDescentVariationParams.append(gradientDescentParam)

# words = ['1','8']
print(words)

def PerformSimpleGradientDescent(xTrainRaw,xValidateRaw,xTestRaw,param):
    (xTrain, xValidate, xTest) = SMSSpamSupport.FeaturizeWithBagOfWords(xTrainRaw,xValidateRaw,xTestRaw,param["words"])
    import LogisticRegressionModel
    model = LogisticRegressionModel.LogisticRegressionModel(featureCount=param["featureCount"])
    PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,param["featureText"])

###################################################################################################
##########Execute gradient descent for frequency based and mutual index based selection############
###################################################################################################
Parallel(n_jobs=-1)(delayed(PerformSimpleGradientDescent)(xTrainRaw, xValidateRaw, xTestRaw, param) for param in gradientDescentVariationParams)

#################################################################################################
#################K-Fold test
##################################################################################
#################################################################################################
(xMessageFolds, yPredictionFolds) = SMSSpamSupport.CreateFolds(xRaw, yRaw, 5)

def PerformFoldTrainingAndTest(xMessageFolds, yPredictionFolds , param):
    print("Cross fold testing:CurrentFold",param["testFold"])
    (xTrainFoldData, yTrainFoldData, xValidationFoldData, yValidationFoldData) = SMSSpamSupport.SelectTrainAndTestFromFolds(xMessageFolds,yPredictionFolds,param["testFold"])

    wordCounts = SMSSpamSupport.TokenizeFeatures(xTrainFoldData)

    if( param["selectionAlgorithm"] == "frequency"):
        words = SMSSpamSupport.GetFeaturesByFrequency(wordCounts, top = 10)
    else:
        words = SMSSpamSupport.GetFeaturesByMutualInformation(xTrainFoldData,yTrainFoldData,wordCounts,top=10)

    (xTrainFoldDataFeaturized, xValidateFoldDataFeaturized) = SMSSpamSupport.FeaturizeWithBagOfWordsForCrossFold(xTrainFoldData,xValidationFoldData, words)

    import LogisticRegressionModel
    model = LogisticRegressionModel.LogisticRegressionModel(param["featureCount"])
    
    message = "{} Testing Fold {}".format(param["featureText"],param["testFold"])
    PerformGradientDescentIteration(model,xTrainFoldDataFeaturized,yTrainFoldData,xValidateFoldDataFeaturized,yValidationFoldData,message)

kFoldParams = []
for i in range(5):
    for selectionAlgorithm in ["frequency","mutualindex"]:
        kFoldParam={}
        kFoldParam["selectionAlgorithm"] = selectionAlgorithm
        kFoldParam["featureCount"]= 10
        kFoldParam["featureText"] = selectionAlgorithm
        kFoldParam["testFold"] = i+1
        kFoldParams.append(kFoldParam)

Parallel(n_jobs=-1)(delayed(PerformFoldTrainingAndTest)(xMessageFolds, yPredictionFolds, param) for param in kFoldParams)

