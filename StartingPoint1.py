import SMSSpamSupport
from EvaluationsStub import TopDeviatingProbabilities
from EvaluationsStub import Accuracy
import EvaluationsStub
import matplotlib.pyplot as plt 
import SMSSpamKaggleSupport
from joblib import Parallel, delayed

def PerformGradientDescentIteration(model , xTrain,yTrain,xValidate,yValidate,iterationTitle):
    for i in range(100):
        model.fit(xTrain, yTrain, iterations=i, step=0.01)
        yValidatePredicted = model.predict(xValidate)
        featureWeight = model.featureWeights[2]
        validationSetLoss = model.loss(xValidate,yValidate)
        validationsetAccuracy = EvaluationsStub.Accuracy(yValidate, yValidatePredicted)
        #print("%d, %f, %f, %f" % (i, featureWeight, validationSetLoss, validationsetAccuracy))

    print(iterationTitle)
    print(EvaluationsStub.ExecuteAll(yValidate, yValidatePredicted))

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
    # print(fpThresholds)

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

###################################################################################################
##########Feature Selection frequency
##########based########################################################
###################################################################################################
wordCounts = SMSSpamSupport.TokenizeFeatures(xTrainRaw)
words = SMSSpamSupport.GetFeaturesByFrequency(wordCounts, top = 10)
print(words)

####################################################################################################
###########Training using high frequency
###########words######################################################
####################################################################################################
#(xTrain, xValidate, xTest) =
#SMSSpamSupport.FeaturizeWithBagOfWords(xTrainRaw,xValidateRaw,xTestRaw,words,addBiasWeight=True)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=10)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"With
#top 10 high frequency words")

###################################################################################################
##########Feature Selection MI
##########based###############################################################
###################################################################################################
words = SMSSpamSupport.GetFeaturesByMutualInformation(xTrainRaw,yTrain,wordCounts,top=100)
words = ['1','8']
print(words)

####################################################################################################
###########Training using top 10 words by MI
###########Index##################################################
####################################################################################################
#(xTrain, xValidate, xTest) =
#SMSSpamSupport.FeaturizeWithBagOfWords(xTrainRaw,xValidateRaw,xTestRaw,words[:10],addBiasWeight=True)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=10)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"With
#top 10 features by mutual index score")
#PlotPRAndRocCurve(model, xValidate, yValidate, filenamePr= "prcurve1.png",
#filenameRoc="roccurve1.png", plotNumber=0,colorLabel='bo-')

####################################################################################################
###########Training using top 10 words by MI Index + handcrafted
###########features###########################
####################################################################################################
#(xTrain, xValidate, xTest) =
#SMSSpamSupport.FeaturizeWithBagOfWords(xTrainRaw,xValidateRaw,xTestRaw,words[:10],addBiasWeight=True,addCustomFeatures=True)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=15)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"With
#top 10 features by mutual index score and hand crafted features")
#PlotPRAndRocCurve(model, xValidate, yValidate, filenamePr= "prcurve2.png",
#filenameRoc="roccurve2.png", plotNumber=0,colorLabel='go-')

###################################################################################################
##########Training using top 100 words by MI Index + handcrafted
##########features###########################
###################################################################################################
#(xTrain, xValidate, xTest ,xKaggle) =
#SMSSpamSupport.FeaturizeWithBagOfWordsForKaggle(xTrainRaw,xValidateRaw,xTestRaw,
#xKaggleRaw, words[:100],addBiasWeight=True,addCustomFeatures=True)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=103)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"With
#top 100 features by mutual index score and hand crafted features")
#yValidatePredictedRaw = model.predictPreThresholding(xValidate)
#topDeviatingWhenY0 =
#EvaluationsStub.TopDeviatingProbabilities(xValidateRaw,yValidate,yValidatePredictedRaw,expectedy=0)
#topDeviatingWhenY1 =
#EvaluationsStub.TopDeviatingProbabilities(xValidateRaw,yValidate,yValidatePredictedRaw,expectedy=1)
#print(topDeviatingWhenY0[:20])
#print(topDeviatingWhenY1[:20])

#yKagglePredictionsPrethresholding = model.predictPreThresholding(xKaggle)
#yKagglePredictions = model.predict(xKaggle)
#SMSSpamKaggleSupport.OutputSubmission("kaggleOutput",xKaggleIds,yKagglePredictions)

###################################################################################################
##########Leave out Accuracy
##########Wrapper########################################################################
###################################################################################################
#(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw,
#xTestRaw, True,skipFeatureIndex=1)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=4)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"skip
#feature 1")

#(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw,
#xTestRaw, True,skipFeatureIndex=2)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=4)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"skip
#feature 2")

#(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw,
#xTestRaw, True,skipFeatureIndex=3)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=4)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"skip
#feature 3")

#(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw,
#xTestRaw, True,skipFeatureIndex=4)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=4)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"skip
#feature 4")

#(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw,
#xTestRaw, True,skipFeatureIndex=5)
#import LogisticRegressionModel
#model = LogisticRegressionModel.LogisticRegressionModel(featureCount=4)
#PerformGradientDescentIteration(model,xTrain,yTrain,xValidate,yValidate,"skip
#feature 5")
