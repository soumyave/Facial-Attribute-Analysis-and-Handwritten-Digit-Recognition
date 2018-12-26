import logisticRegression as ml1
import hiddenLayerMLP as ml2
import convolutednn as ml3
import uspsImages as usps
from tensorflow.examples.tutorials.mnist import input_data      

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
uspsData,uspsLabels=usps.preprocess()
print("UBitName = himasuja")
print("personNumber = 50246828")
ml1.train_and_test_LR(mnist,uspsData,uspsLabels)
ml2.train_and_test_MLP(mnist,uspsData,uspsLabels)
ml3.train_and_test_convNN(mnist,uspsData,uspsLabels)