import collections
import os

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 31561

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        rawFields = l.split(",")
        if(len(rawFields) != 15):
            message = "Expected 15 fields, found %d in: %s" % (len(fields), str(l))
            raise UserWarning(message)

        fields = [ x.strip() for x in rawFields ]

        if(fields[-1] == "<=50K"):
            y.append(0)
            x.append(fields[0:-1])
        elif(fields[-1] == ">50K"):
            y.append(1)
            x.append(fields[0:-1])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

def TrainValidateTestSplit(x, y, percentValidate = .1, percentTest = .1):
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
    yValidate = y[numTest: numTest+numValidate]
    yTrain = y[numTest+numValidate:]

    return (xTrain, yTrain, xValidate, yValidate, xTest, yTest)


def GetValues(dataSet, featureID):
    values = []
    for x in dataSet:
        if x[featureID] not in values:
            values.append(x[featureID])
    return values

def OneHotFeaturesFor(x, i, valuesTable):
    encoding = []
    for value in valuesTable:
        encoding.append(1 if x[i] == value else 0)

    return encoding


def DoFeaturize(xRaw, includeInNumeric=None, oneHotValuesTable = None):
    dataSet = []

    for x in xRaw:
        features = []

        # Basic use of the available features

        if oneHotValuesTable != None:
            for i in range(len(oneHotValuesTable)):
                features = features + OneHotFeaturesFor(x, i, oneHotValuesTable[i])
                

        if includeInNumeric != None:
            for i in range(len(includeInNumeric)):
                if includeInNumeric[i]:
                    features.append(int(x[i]))

        dataSet.append(features)

    return dataSet


def Featurize(xTrainRaw, xValidateRaw, xTestRaw):
    
    # featurize the training data.
    includeInOneHot  = [False,True ,False,True ,False,True ,True ,True ,True ,False,False,False,False,True ]
    includeInNumeric = [True ,False,False,False,True ,False,False,False,False,False,True ,True ,True ,False]

    oneHotValuesTable = []
    for i in range(len(includeInOneHot)):
        if includeInOneHot[i]:
            oneHotValuesTable.append(GetValues(xTrainRaw, i))
        else:
            oneHotValuesTable.append([])

    xTrain = DoFeaturize(xTrainRaw, includeInNumeric=includeInNumeric, oneHotValuesTable=oneHotValuesTable)
    
    # now featurize validate and test using any features discovered on the training set. Don't use the validate or test set to influence which features to use.
    xValidate = DoFeaturize(xValidateRaw, includeInNumeric=includeInNumeric, oneHotValuesTable=oneHotValuesTable)
    xTest = DoFeaturize(xTestRaw, includeInNumeric=includeInNumeric, oneHotValuesTable=oneHotValuesTable)

    return (xTrain, xValidate, xTest)

def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])
