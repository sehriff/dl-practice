from NeuralNetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 2, 1], 'tanh')
x = np.array([[1,0],[0,0],[0,1],[1,1]])
y = np.array([1,0,1,0])
nn.fit(x,y)
for i in [[1,0],[0,0],[0,1],[1,1]]:
    print(i, nn.predict(i))