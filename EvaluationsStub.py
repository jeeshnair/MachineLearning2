import math

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    matrix = ConfusionMatrix(y, yPredicted)
    # (# TP + # TN) / # Total
    return (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])

def ErrorBounds(y,yPredicted):
    accuracy = Accuracy(y, yPredicted)
    upperBound= accuracy + (1.96 * math.sqrt((accuracy*(1-accuracy))/len(y)))
    lowerBound= accuracy - (1.96 * math.sqrt((accuracy*(1-accuracy))/len(y)))

    return (upperBound, lowerBound)

def Precision(y, yPredicted):
    matrix = ConfusionMatrix(y, yPredicted)
    # TP / (# TP + # FP)
    if matrix[0][0] + matrix[1][0] == 0:
        return 0
    return (matrix[0][0]) / (matrix[0][0] + matrix[1][0])

def Recall(y, yPredicted):
    matrix = ConfusionMatrix(y, yPredicted)
    # TP / (# TP + # FN)
    return (matrix[0][0]) / (matrix[0][0] + matrix[0][1])

def FalseNegativeRate(y, yPredicted):
    matrix = ConfusionMatrix(y, yPredicted)
    # FN / (# TP + # FN)
    return (matrix[0][1]) / (matrix[0][0] + matrix[0][1])

def FalsePositiveRate(y, yPredicted):
    matrix = ConfusionMatrix(y, yPredicted)
    #  FP / (# FP + # TN)
    return (matrix[1][0]) / (matrix[1][0] + matrix[1][1])

def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    matrix = [[0 for i in range(2)] for i in range(2)]
    for i in range(len(y)):
        if y[i] == 1:
            # True positive
            if yPredicted[i] == 1:
                matrix[0][0] = matrix[0][0] + 1
            # False Negative
            if yPredicted[i] == 0:
                matrix[0][1] = matrix[0][1] + 1
        else:
            # False Positive
            if yPredicted[i] == 1:
                matrix[1][0] = matrix[1][0] + 1
            # True Negative
            if yPredicted[i] == 0:
                matrix[1][1] = matrix[1][1] + 1
            
    return matrix

def PrintConfusionMatrix(confusionMatrix):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in confusionMatrix]))

def ExecuteAll(y, yPredicted):
    confusionMatrix = ConfusionMatrix(y, yPredicted)
    PrintConfusionMatrix(confusionMatrix)
    # print(confusionMatrix)
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    print("Error Bounds:",ErrorBounds(y, yPredicted))

def TopDeviatingProbabilities(x, y, yPredictedRaw, expectedy):
    fps = []
    for i in range(len(y)):
        if(y[i] == expectedy):
            fp = []
            fp.append(x[i])
            fp.append(yPredictedRaw[i])
            fps.append(fp)

    fps.sort(key = TakeRawProbability,reverse = True)
    return fps

def TakeRawProbability(element):
    return element[1]
