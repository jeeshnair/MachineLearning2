import SMSSpamSupport
from EvaluationsStub import TopDeviatingProbabilities
from EvaluationsStub import Accuracy
import EvaluationsStub
import matplotlib.pyplot as plt 
import SMSSpamKaggleSupport
from joblib import Parallel, delayed

def PerformGradientDescentIteration(model , xTrain,yTrain,xValidate,yValidate,param):
    for i in range(10):
        model.fit(xTrain, yTrain, iterations=i, step=0.01)

    yValidatePredicted = model.predict(xValidate)
    print(param)
    print(EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted))
    print("Loss",model.loss(xValidate,yValidate))

def PlotPrecisionRecall(precision, recall , fileName, plotNumber,colorLabel):
    # plotting the points
    plt.figure(plotNumber,figsize=(10,10))
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision,colorLabel)  
    plt.show(block=False)
    plt.savefig(fileName)

def PlotRocCurve(fp, fn , fileName, plotNumber,colorLabel):
    # plotting the points
    plt.figure(plotNumber)
    plt.xlabel("False Positive")
    plt.ylabel("False Negative")
    plt.plot(fp, fn,colorLabel)  
    plt.show(block=False)
    plt.savefig(fileName)

def PlotPRAndRocCurve(model, xValidate, yValidate, filenamePr,filenameRoc, plotNumber,colorLabel):
    precision = []
    recall = []
    fps = []
    fns = []
    fpThresholds = []
    for i in range(100):
        yValidatePredicted = model.predict(xValidate,(i + 1) / 100)
        precision.append(EvaluationsStub.Precision(yValidate,yValidatePredicted))
        recall.append(EvaluationsStub.Recall(yValidate,yValidatePredicted))

        fp = EvaluationsStub.FalsePositiveRate(yValidate,yValidatePredicted)
        fn = EvaluationsStub.FalseNegativeRate(yValidate,yValidatePredicted)

        fpThreshold = []
        fpThreshold.append(i)
        fpThreshold.append(fp)
        fpThreshold.append(fn)
        fpThresholds.append(fpThreshold)

        fps.append(fp)
        fns.append(fn)

    PlotPrecisionRecall(precision, recall, filenamePr, plotNumber, colorLabel)
    PlotRocCurve(fps,fns,filenameRoc, plotNumber + 1,colorLabel)

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

kDataPath = "SMSSpamCollection"
kaggleDataPath = "SMSSpamCollection_test"

(xKaggleRaw, xKaggleIds) = SMSSpamKaggleSupport.LoadKaggleData(kaggleDataPath)

gradientDescentVariationParams = []

param = {}
param["includeCapitilizationFeature"] = False
param["wordCount"] = 100
param["includeCurrencySymbolFeature"] = False
param["featureCount"] = 105
param["featureText"] = "100 mi words + custom features + split words"
param["includeCustomBaselineFeatures"] = True
param["outputFile"] = "Kaggle1"
param["prcurveFile"] = "PRCurve1"
param["roccurveFile"] = "ROCCurve1"
param["plotNumber"] = 0
param["colorLabel"] = "go-"
param["splitwords"] = True
param["caseInsensitive"] = False

gradientDescentVariationParams.append(param)

param = {}
param["includeCapitilizationFeature"] = False
param["wordCount"] = 100
param["includeCurrencySymbolFeature"] = False
param["featureCount"] = 105
param["featureText"] = "100 mi words + custom features"
param["includeCustomBaselineFeatures"] = True
param["outputFile"] = "Kaggle2"
param["prcurveFile"] = "PRCurve2"
param["roccurveFile"] = "ROCCurve2"
param["plotNumber"] = 0
param["colorLabel"] = "go-"
param["splitwords"] = False
param["caseInsensitive"] = False

gradientDescentVariationParams.append(param)

param = {}
param["includeCapitilizationFeature"] = True
param["wordCount"] = 100
param["includeCurrencySymbolFeature"] = False
param["featureCount"] = 106
param["featureText"] = "100 mi words + custom features+ caseinsensitive "
param["includeCustomBaselineFeatures"] = True
param["outputFile"] = "Kaggle3"
param["prcurveFile"] = "PRCurve3"
param["roccurveFile"] = "ROCCurve3"
param["plotNumber"] = 0
param["colorLabel"] = "go-"
param["splitwords"] = False
param["caseInsensitive"] = True

gradientDescentVariationParams.append(param)

def PerformSimpleGradientDescent(xTrainRaw,xValidateRaw,xTestRaw,xKaggleRaw, yTrain,param):
    wordCounts = SMSSpamSupport.TokenizeFeatures(xTrainRaw)
    words = SMSSpamSupport.GetFeaturesByMutualInformation(xTrainRaw,yTrain,wordCounts,top=param["wordCount"])
    print(words)
    import LogisticRegressionModel
    model = LogisticRegressionModel.LogisticRegressionModel(featureCount=param["featureCount"])
    (xTrain, xValidate, xTest ,xKaggle) = SMSSpamSupport.FeaturizeWithBagOfWordsForKaggle(xTrainRaw,xValidateRaw,xTestRaw,xKaggleRaw,words,param)
    PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,param)
    yKagglePredictions = model.predict(xKaggle)
    PlotPRAndRocCurve(model, xValidate, yValidate, param["prcurveFile"],param["roccurveFile"], param["plotNumber"],param["colorLabel"])
    SMSSpamKaggleSupport.OutputSubmission(param["outputFile"],xKaggleIds,yKagglePredictions)

Parallel(n_jobs=-1)(delayed(PerformSimpleGradientDescent)(xTrainRaw, xValidateRaw, xTestRaw, xKaggleRaw, yTrain, param) for param in gradientDescentVariationParams)