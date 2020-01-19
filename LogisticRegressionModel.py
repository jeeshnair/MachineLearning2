import math

class LogisticRegressionModel(object):

    # default to 5
    def __init__(self, featureCount=5):
        self.featureWeights = [0 for i in range(featureCount + 1)]

    def fit(self, x, y, iterations, step):
        predictions = self.predictPreThresholding(x)

        gradient = [0 for i in range(len(self.featureWeights))]
        for i in range(len(x)):
            for j in range(len(x[i])):
                gradient[j] = gradient[j] + (predictions[i] - y[i]) * x[i][j]
                
        for i in  range(len(gradient)):
             self.featureWeights[i] = self.featureWeights[i] - (step*(gradient[i] / len(x)))
        
    def loss(self, x, y):
        predictions = self.predictPreThresholding(x)   
        datasetLoss = 0
        for i in range(len(y)):
            datasetLoss = datasetLoss + (-y[i] * math.log(predictions[i]) - (1 - y[i]) * math.log(1 - predictions[i]))

        return datasetLoss / len(y)

    def predict(self, x, threshold=0.5):
        predictions = []
        thresholdPredictions = []

        for i in range(len(x)):
            featureWeightProductSum = 0
            for j in range(len(x[i])):
                featureWeightProductSum = featureWeightProductSum + (x[i][j] * self.featureWeights[j])

            prediction = 1 / (1 + math.exp(-1 * (featureWeightProductSum)))
            if prediction >= threshold:
                thresholdPredictions.append(1)
            else:
                thresholdPredictions.append(0)

        return thresholdPredictions

    def predictPreThresholding(self , x):
        predictions = []

        for i in range(len(x)):
            featureWeightProductSum = 0
            for j in range(len(x[i])):
                featureWeightProductSum = featureWeightProductSum + (x[i][j] * self.featureWeights[j])

            prediction = 1 / (1 + math.exp(-1 * (featureWeightProductSum)))
            predictions.append(prediction)

        return predictions