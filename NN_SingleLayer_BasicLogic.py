
# coding: utf-8

# In[ ]:


import numpy as np
import time
import numpy.random as rand
import matplotlib.pyplot as plt


# In[ ]:


class NeuralNetwork:
    import numpy as np
    
    def __init__ (self, trainingInputs, trainingOutputs):
        
        self.trainingInputs = trainingInputs
        self.trainingOutputs = trainingOutputs
        
        inputLength = self.trainingInputs.shape[1]
        
        rand.seed(int(time.time()))
        self.inputWeights = (rand.random((inputLength, 1)) * 2) - 1
        
    def __sigmoid__ (self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoidDerivative__ (self, x):
        return x * (1 - x)
    
    def __calculateRMSE__ (self, error):
        MSE = (error ** 2).sum()
        return np.sqrt(MSE)
    
    def answer (self, input):
        rawOutput = input.dot(self.inputWeights)
        return self.__sigmoid__(rawOutput)
    
    def train (self, iterations = 10000):
        RMSEs = np.zeros(iterations)
        
        plt.clf()
        for i in range(0, iterations):
            outputs = self.answer(self.trainingInputs)
            errors = self.trainingOutputs - outputs
            RMSEs[i] = self.__calculateRMSE__(errors)
            
            adjustWeight = self.trainingInputs.T.dot(errors * self.__sigmoidDerivative__(outputs))
            self.inputWeights += adjustWeight
            
        plt.xscale('log')
        plt.plot(np.array(range(0, iterations)) + 1, RMSEs)
        plt.show()        


# In[ ]:


inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

outputs = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T


# In[ ]:


nn = NeuralNetwork(inputs, outputs)
nn.train(10000)


# In[ ]:


opt = nn.answer(inputs).round()
print(opt == outputs)

