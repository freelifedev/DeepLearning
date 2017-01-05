# import the necessary pckages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn import datasets

# load the MINST dataset and split it into training and testing data
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(mnist.data, mnist.target,
                                                                  test_size=0.5, random_state=42)

# train the perceptron
model = Perceptron(n_iter=30, eta0=1.0, random_state=84)
model.fit(trainData, trainLabels)

# evaluate the Perceptron
predictions = model.predict(testData)

print("data = {}, type = {}".format(mnist.target_names, type(mnist.target_names[0])))
print(" string = {}".format(mnist.target_names.astype(str)))


print(classification_report(predictions, testLabels, target_names=mnist.target_names.astype(str)))
