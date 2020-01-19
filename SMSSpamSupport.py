import collections
import os
import math

def CreateFolds(xMessages,yPredictions, foldCount):
    currentFold = 0
    xMessageFolds = [[] for i in range(foldCount)]
    for x in xMessages:
        if(currentFold == foldCount):
            currentFold = 0
        xMessageFolds[currentFold].append(x)
        currentFold = currentFold + 1

    currentFold = 0
    yPredictionFolds = [[] for i in range(foldCount)]
    for x in yPredictions:
        if(currentFold == foldCount):
            currentFold = 0
        yPredictionFolds[currentFold].append(x)
        currentFold = currentFold + 1

    return (xMessageFolds, yPredictionFolds)

def SelectTrainAndTestFromFolds(xMessageFolds, yPredictionFolds, testFoldIndex):
    xTrain = []
    xTest = []
    yTrainPrediction = []
    yTestPrediction = []
    for i in range(len(xMessageFolds)):
        if i != testFoldIndex - 1:
            xTrain.extend(xMessageFolds[i])
            yTrainPrediction.extend(yPredictionFolds[i])
        else:
            xTest.extend(xMessageFolds[i])
            yTestPrediction.extend(yPredictionFolds[i])
    
    return (xTrain, yTrainPrediction, xTest, yTestPrediction)
    
def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5474

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

def TrainValidateTestSplit(x, y, percentValidate=.1, percentTest=.1):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)
    numValidate = round(len(x) * percentValidate)

    if(numValidate == 0 or numValidate > len(y)):
        raise UserWarning("Attempting to split into training, validation and testing set.\n\tSome problem with the percentValidate or data set size. Check your work and try again.")

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training, validation and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xValidate = x[numTest:numTest + numValidate]
    xTrain = x[numTest + numValidate:]
    yTest = y[:numTest]
    yValidate = y[numTest: numTest + numValidate]
    yTrain = y[numTest + numValidate:]

    return (xTrain, yTrain, xValidate, yValidate, xTest, yTest)


def DoFeaturize(xRaw, addBiasWeight,skipFeatureIndex):
    words = ['call', 'to', 'your', 'sms','sex']
    dataSet = []

    print("Skipping Feature {} !".format(skipFeatureIndex))
    if(skipFeatureIndex == 3):
        words.remove(words[0])
            
    if(skipFeatureIndex == 4):
        words.remove(words[1])

    if(skipFeatureIndex == 5):
        words.remove(words[2])

    for x in xRaw:
        features = []

        if addBiasWeight == True:
            features.append(1)

        if(skipFeatureIndex == 1):
            pass
        else:
            # Have a feature for longer texts
            if(len(x) > 40):
                features.append(1)
            else:
                features.append(0)

        if(skipFeatureIndex == 2):
            pass
        else:
            # Have a feature for texts with numbers in them
            if(any(i.isdigit() for i in x)):
                features.append(1)
            else:
                features.append(0)
        
        # Have features for a few words
        for word in words:
            if word in x:
                features.append(1)
            else:
                features.append(0)

        dataSet.append(features)

    return dataSet

def Featurize(xTrainRaw, xValidateRaw, xTestRaw, addBiasWeight, skipFeatureIndex=-1):
    
    # featurize the training data.
    xTrain = DoFeaturize(xTrainRaw, addBiasWeight,skipFeatureIndex)
    
    # now featurize validate and test using any features discovered on the
    # training set.  Don't use the validate or test set to influence which
    # features (e.g.  words) to use.
    xValidate = DoFeaturize(xValidateRaw, addBiasWeight,skipFeatureIndex)
    xTest = DoFeaturize(xTestRaw, addBiasWeight,skipFeatureIndex)

    return (xTrain, xValidate, xTest)

def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])

def FeaturizeWithBagOfWords(xTrainRaw, xValidateRaw, xTestRaw, words, addBiasWeight=True,addCustomFeatures=False):
    xTrain = DoFeaturizeWithBagOfWords(xTrainRaw, words, addBiasWeight, addCustomFeatures)
    xValidate = DoFeaturizeWithBagOfWords(xValidateRaw, words,addBiasWeight, addCustomFeatures)
    xTest = DoFeaturizeWithBagOfWords(xTestRaw, words,addBiasWeight, addCustomFeatures)

    return (xTrain, xValidate, xTest)

def FeaturizeWithBagOfWordsForCrossFold(xTrainRaw, xValidateRaw, words, addBiasWeight=True,addCustomFeatures=False):
    xTrain = DoFeaturizeWithBagOfWords(xTrainRaw, words, addBiasWeight, addCustomFeatures)
    xValidate = DoFeaturizeWithBagOfWords(xValidateRaw, words,addBiasWeight, addCustomFeatures)

    return (xTrain, xValidate)

def FeaturizeWithBagOfWordsForKaggle(xTrainRaw, xValidateRaw, xTestRaw,xKaggleRaw, words, param):
    xTrain = DoFeaturizeWithBagOfWords(xTrainRaw, words,addBiasWeight= True, addCustomFeatures= param["includeCustomBaselineFeatures"],addCaptizationFeature=param["includeCapitilizationFeature"],splitWords=param["splitwords"],caseInsensitive=param["caseInsensitive"])
    xValidate = DoFeaturizeWithBagOfWords(xValidateRaw, words,addBiasWeight= True, addCustomFeatures= param["includeCustomBaselineFeatures"],addCaptizationFeature=param["includeCapitilizationFeature"],splitWords=param["splitwords"],caseInsensitive=param["caseInsensitive"])
    xTest = DoFeaturizeWithBagOfWords(xTestRaw, words,addBiasWeight= True, addCustomFeatures= param["includeCustomBaselineFeatures"],addCaptizationFeature=param["includeCapitilizationFeature"],splitWords=param["splitwords"],caseInsensitive=param["caseInsensitive"])
    xKaggle = DoFeaturizeWithBagOfWords(xKaggleRaw, words,addBiasWeight= True, addCustomFeatures= param["includeCustomBaselineFeatures"],addCaptizationFeature=param["includeCapitilizationFeature"],splitWords=param["splitwords"],caseInsensitive=param["caseInsensitive"])

    return (xTrain, xValidate, xTest,xKaggle)

def DoFeaturizeWithBagOfWords(xRaw, words,addBiasWeight,addCustomFeatures=False,addCaptizationFeature=False,addCurrencyFeatures=False,splitWords=False,caseInsensitive=False):
    dataSet = []

    if(addCustomFeatures == True):
        if('call' not in words):
            words.append('call')
        if('to' not in words):
            words.append('to')
        if('your' not in words):
            words.append('your')

    for x in xRaw:
        features = []

        if addBiasWeight == True:
            features.append(1)

        if(addCustomFeatures == True):
            if(len(x) > 40):
                features.append(1)
            else:
                features.append(0)

            if(any(i.isdigit() for i in x)):
                features.append(1)
            else:
                features.append(0)

        # Have features for a few words
        for word in words:
            if(splitWords == False):
                if(caseInsensitive == True):
                    if word.lower() in x.lower():
                        features.append(1)
                    else:
                        features.append(0)
                else:
                    if word in x:
                        features.append(1)
                    else:
                        features.append(0)
            else:
                if(caseInsensitive == True):
                    wordsInMessage = x.split()
                    for wordInMessage in wordsInMessage:
                        if word.lower() == wordInMessage.lower():
                            features.append(1)
                        else:
                            features.append(0)
                else:
                    if word in x.split():
                        features.append(1)
                    else:
                        features.append(0)

        if(addCaptizationFeature == True):
            upperCaseWordCount = 0
            wordsInMessage = x.split()
            for word in wordsInMessage:
                if(word.isupper() and len(word) > 2):
                    upperCaseWordCount = upperCaseWordCount + 1 

            if(upperCaseWordCount > 1):
                features.append(1)
            else:
                features.append(0)

        dataSet.append(features)

    return dataSet

def TokenizeFeatures(xTrainRaw):
    words = []
    count = []
    for x in xTrainRaw:
        splitWords = x.split()
        for word in splitWords:
            if word not in words:
                words.append(word)
                wordCount = []
                wordCount.append(word)
                wordCount.append(1)
                count.append(wordCount)
            else:
                count[words.index(word)][1] = count[words.index(word)][1] + 1

    return count

def GetFeaturesByFrequency(wordCounts,top):
    highFrequencyWords = []
    wordCounts.sort(key = TakeCount, reverse = True)
    for i in range(top):
        highFrequencyWords.append(wordCounts[i][0])

    return highFrequencyWords

def TakeCount(list):
    return list[1]

def GetFeaturesByMutualInformation(xTrainRaw,yTrain, wordCounts, top):
    miIndexes = []
    topmiIndexes = []

    for word in wordCounts:
        miIndexRow = []
        contingencyMatrix = [[0 for i in range(2)] for i in range(2)]
        for i in range(len(xTrainRaw)):
            if word[0] in xTrainRaw[i]:
                if(yTrain[i] == 1):
                    contingencyMatrix[1][1] = contingencyMatrix[1][1] + 1
                else:
                    contingencyMatrix[0][1] = contingencyMatrix[0][1] + 1
            else:
                if(yTrain[i] == 1):
                    contingencyMatrix[1][0] = contingencyMatrix[1][0] + 1
                else:
                    contingencyMatrix[0][0] = contingencyMatrix[0][0] + 1
        
        x0y0Probability = (contingencyMatrix[0][0] + 1) / (len(xTrainRaw) + 2)
        x0Probability = (contingencyMatrix[0][0] + contingencyMatrix[1][0] + 1) / (len(xTrainRaw) + 2)
        y0Probability = (contingencyMatrix[0][0] + contingencyMatrix[0][1] + 1) / (len(xTrainRaw) + 2)
        miIndex = x0y0Probability * math.log(x0y0Probability / (x0Probability * y0Probability))

        x0y1Probability = (contingencyMatrix[1][0] + 1) / (len(xTrainRaw) + 2)
        x0Probability = (contingencyMatrix[0][0] + contingencyMatrix[1][0] + 1) / (len(xTrainRaw) + 2)
        y1Probability = (contingencyMatrix[1][0] + contingencyMatrix[1][1] + 1) / (len(xTrainRaw) + 2)
        miIndex = miIndex + (x0y1Probability * math.log(x0y1Probability / (x0Probability * y1Probability)))

        x1y0Probability = (contingencyMatrix[0][1] + 1) / (len(xTrainRaw) + 2)
        x1Probability = (contingencyMatrix[0][1] + contingencyMatrix[1][1] + 1) / (len(xTrainRaw) + 2)
        y0Probability = (contingencyMatrix[0][0] + contingencyMatrix[0][1] + 1) / (len(xTrainRaw) + 2)
        miIndex = miIndex + (x1y0Probability * math.log(x1y0Probability / (x1Probability * y0Probability)))

        x1y1Probability = (contingencyMatrix[1][1] + 1) / (len(xTrainRaw) + 2)
        x1Probability = (contingencyMatrix[0][1] + contingencyMatrix[1][1] + 1) / (len(xTrainRaw) + 2)
        y1Probability = (contingencyMatrix[1][0] + contingencyMatrix[1][1] + 1) / (len(xTrainRaw) + 2)
        miIndex = miIndex + (x1y1Probability * math.log(x1y1Probability / (x1Probability * y1Probability)))
        
        miIndexRow.append(word[0])
        miIndexRow.append(miIndex)
        miIndexes.append(miIndexRow)

    miIndexes.sort(key = TakeCount, reverse = True)

    for i in range(top):
        topmiIndexes.append(miIndexes[i][0])

    return topmiIndexes