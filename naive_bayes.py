import math
import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def separateByClass(self, X, y):
        separated = [[] for i in range(self.num_class)]
        for data, true_label in zip(X, y):
            separated[true_label].append(data)
        return separated

    def summarize(self, X):
        X = np.array(X)
        mean_feature = X.mean(0)
        var_feature = X.var(0)
        summaries = [[mean, var] for mean, var in zip(mean_feature, var_feature)]
        return summaries

    def summarizeByClass(self, X, y):
        separated = self.separateByClass(X, y)
        summaries = [[] for i in range(self.num_class)]
        for i in range(self.num_class):
            summaries[i] = self.summarize(separated[i])
        return summaries

    def calGaussianProb(self, x, mean, var):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*var)))
        return (1 / math.sqrt(2*math.pi * var)) * exponent

    def getPredictions(self, x):
        class_prob = []
        for i in range(self.num_class):
            prob = 1
            for j, (mean, var) in enumerate(self.summaries[i]):
                prob *= self.calGaussianProb(x[j], mean, var)
            class_prob.append(prob)
        return np.argmax(class_prob)

    def fit(self, X, y):
        assert len(X) == len(y)
        num_class = len(set(y))
        self.num_class = num_class
        self.summaries = self.summarizeByClass(X, y)
        return self

    def predict(self, X):
        y_predict = []
        for x in X:
            y_predict.append(self.getPredictions(x))
        return y_predict