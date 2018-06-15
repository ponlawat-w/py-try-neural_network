
# coding: utf-8

# In[44]:


import numpy as np
import time
import numpy.random as rand
import matplotlib.pyplot as plt


# In[45]:


class NeuralLayer:
    
    def __init__ (self, neuronAmount, neuronInputAmount):
        rand.seed(int(time.time()))
        self.weights = 2 * rand.random((neuronInputAmount, neuronAmount))


# In[46]:


class NeuralNetwork:
    import numpy as np
    
    def __init__ (self, trainingInputs, trainingOutputs, layer1, layer2):
        
        self.trainingInputs = trainingInputs
        self.trainingOutputs = trainingOutputs
        self.layer1 = layer1
        self.layer2 = layer2
        
    def __sigmoid (self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoidDerivative (self, x):
        return x * (1 - x)
    
    def __calculateRMSE (self, error):
        MSE = (error ** 2).sum()
        return np.sqrt(MSE)
    
    def answer (self, input):
        layer1RawOutput = input.dot(self.layer1.weights)
        layer1Output = self.__sigmoid(layer1RawOutput)
        
        layer2RawOutput = layer1Output.dot(self.layer2.weights)
        layer2Output = self.__sigmoid(layer2RawOutput)
        return (layer1Output, layer2Output)
    
    def train (self, iterations = 10000):
        layer1RMSEs = np.zeros(iterations)
        layer2RMSEs = np.zeros(iterations)
        
        plt.clf()
        for i in range(0, iterations):
            (layer1Outputs, layer2Outputs) = self.answer(self.trainingInputs)
            
            layer2Error = self.trainingOutputs - layer2Outputs
            layer2Delta = layer2Error * self.__sigmoidDerivative(layer2Outputs)
            
            layer1Error = layer2Delta.dot(self.layer2.weights.T)
            layer1Delta = layer1Error * self.__sigmoidDerivative(layer1Outputs)
            
            layer1RMSEs[i] = self.__calculateRMSE(layer1Error)
            layer2RMSEs[i] = self.__calculateRMSE(layer2Error)
            
            layer1AdjustWeights = self.trainingInputs.T.dot(layer1Delta)
            layer2AdjustWeights = layer1Outputs.T.dot(layer2Delta)
            
            self.layer1.weights += layer1AdjustWeights
            self.layer2.weights += layer2AdjustWeights
            
        plt.xscale('log')
        plt.plot(np.array(range(0, iterations)) + 1, layer1RMSEs)
        plt.plot(np.array(range(0, iterations)) + 1, layer2RMSEs)
        plt.legend(['Layer 1', 'Layer 2'])
        plt.show()
        
    def printWeights (self):
        print('Layer #1:')
        print(self.layer1.weights)
        
        print('Layer #2:')
        print(self.layer2.weights)


# In[47]:


inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
#    [1, 1, 0],
    [1, 1, 1]
])

#outputs = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T
outputs = np.array([[0, 0, 1, 1, 1, 1, 0]]).T


# In[48]:


l1 = NeuralLayer(4, 3)
l2 = NeuralLayer(1, 4)

nn = NeuralNetwork(inputs, outputs, l1, l2)
nn.train(10000)


# In[49]:


testCases = np.array([
    [1, 1, 0]
])

opt1, opt2 = nn.answer(testCases)
opt2 = opt2.round().astype(int)

print(opt2)
print(opt2 == np.array([[0]]))


# In[50]:


opt1, opt2 = nn.answer(inputs)
opt2 = opt2.round().astype(int)
print(opt2)
print(opt2 == outputs)


# In[51]:


nn.printWeights()

