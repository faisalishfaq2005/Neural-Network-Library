import numpy as np

class DenseLayer:
    def __init__(self,input_neurons,neurons):
        self.input_neurons=input_neurons
        self.neurons=neurons
        
        if input_neurons!= None:
            self.weights=np.random.rand(self.input_neurons,self.neurons)
            self.bias=np.random.rand(1,self.neurons)


    def forward(self,input_data):
        self.dense_input_data=input_data
        self.dense_output_data=np.dot(self.dense_input_data,self.weights)+self.bias
        return self.dense_output_data
    
    
        
