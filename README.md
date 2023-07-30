<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


# 6 Hidden layer Artificial Neural Network from scratch with numpy only (no for loops for back-propagation)

# Hello folks! This notion page is dedicated to showcase how to create a 6 hidden layer artificial neural network to be used in a classification task using vectorisation approach.

## Alternatively, can access the HTML version of this markdown file https://leeyz888.github.io/6-hidden-layer/

## Table of contents

## 1) Dataset description

- MNIST Dataset , commonly used to demonstrate creation of ANNs.
- Comprises billions of handwritten numbers, a small subset is used for demo in sklearn library
- Training dataset is preprocessed using min max scaler to facilitate the learning process
- Train-Dev-Test scheme is used to split data into training, development and testing datasets.

## 2) Network Architecture and derivation of forward pass and back-propagation , using matrix algebra and term-by-term differentiation.

## Network Architecture , with the matrices for each layer.

![Screenshot 2022-08-26 at 2.30.33 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_2.30.33_PM.png)

- Activation is RelU all the way until the final layer, of which the corresponding activation is Softmax.

## Forward pass

![6 Layer Nn -12.jpeg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/6_Layer_Nn_-12.jpeg)

- Or simply, in matrix notation $Z_1=W_{input\ to\ first\ hidden\ layer}*(X)+B_{input\ to\ first\ hidden\ layer}$ of which X is the training data transformed into a matrix of dimensions (number of features)X(number of samples).
- $A_1=ReLU(Z_1)$, which is also the output from the input layer to the first hidden layer.
- This can be generalised to all hidden layers as well as the output layer, i.e $Z_{i+1}=W_{ith\ hidden\ layer\  to\ (i+1)th\ hidden\ layer}*(A_i)+B_{ith\ to\ (i+1)th\ hidden\ layer}$
- And that $A_{i+1}=ReLU(Z_{i+1})$
- In other words:

![6 Layer Nn -13.jpeg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/6_Layer_Nn_-13.jpeg)

- And at the output layer $A_{output}=Softmax(Z_{output})$
- The horizontal lines indicate that the inputs are row vectors, with M columns .

## Back-propagation derivation

### Instead of the usual matrix differentiation which Andrew Ng and several others used, I plan to instead showcase the derivation of the back-propagation formulas using term by term partial differentiation in accordance with the multivariate version of the chain rule.

<aside>
💡 IMO using the multivariable chain rule is more beginner friendly and albeit more verbosic, is more easily understood by normal undergrad students.

</aside>

- Firstly, I plan to write out the formula for the loss function as well as the derivative with respect to $Z_{output}$ or $\hat{y}_{nm}$.
- The loss function is as follows:

![Screenshot 2022-08-26 at 3.03.54 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_3.03.54_PM.png)

where M is the number of samples and N is the number of outputs. $y_{nm}$  in this case is the true value in contrast to the prediction which is $\hat{y}_{nm}$. In other words, the loss for this ANN is the mean categorical-cross entropy. 

- M= 1439 whilst N=10, implying that we plan to classify the picture as showing numbers from 0 to 9.

## Derivative with respect to the predicted output before softmax activation, i.e $Z_7^{O,M}$ of which O denotes the output neuron, and M denotes the index of the input data.

![6 Layer Nn -7.jpg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/6_Layer_Nn_-7.jpg)

![6 Layer Nn -8.jpg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/6_Layer_Nn_-8.jpg)

- As the binary labels (only one 1, rest are 0s) sum up to 1.
- Hence:

![Screenshot 2022-08-26 at 4.30.16 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_4.30.16_PM.png)

of which m denotes the **mth** training data. 

- Thusly,

![Screenshot 2022-08-26 at 4.42.53 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_4.42.53_PM.png)

where : 

![Screenshot 2022-08-26 at 4.46.10 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_4.46.10_PM.png)

## Derivation of dW and dB, i.e the gradients of the Weights and Biases for the 7th layer

- For the bias vector it is suprisingly direct, since the terms are linear per se.
- The image bellow shows the derivation:

![Screenshot 2022-08-26 at 4.50.56 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_4.50.56_PM.png)

<aside>
🌠 The derivative w.r.t the bias term cancels out to 1, hence the gradient for the bias from the 7th layer to the output node is simply the sum of the errors for all samples. The o and the m denote the oth and the mth output node and sample data, respectively.

</aside>

### For dW however it’s a bit more convuluted. Bellow I will try to clearly explain the derivation through term by term partial diffrentiation.

![Screenshot 2022-08-26 at 5.20.09 PM.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Screenshot_2022-08-26_at_5.20.09_PM.png)

![6 Layer Nn -18.jpeg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/6_Layer_Nn_-18.jpeg)

which leaves us with dz6 and etc. 

<aside>
💡 Do note that $A_0=X$ or the input data with dimensions (NXM) as this is the input data for the first layer.

</aside>

## Derivation of dz6 ( and for all $dz_i$ with 0≤i<6)

![截屏2022-08-26 下午5.41.49.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-26_%25E4%25B8%258B%25E5%258D%25885.41.49.png)

## In summary:

![IMG_151545E26F77-1.jpeg](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/IMG_151545E26F77-1.jpeg)

# 3) Implementation of ANN in Python, as well as training on MNIST dataset.

## Neural network class written in Python.

```python
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax as sm
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from math import sqrt
from math import log

class NeuralNet:
    def __init__(self, num_features, num_hidden1,num_hidden2,num_hidden3,num_hidden4,num_hidden5,num_hidden6 ,alpha, max_epochs, num_output, _EPSILON):
        super().__init__()
        self.num_features=num_features  # number of input nodes (features)
        self.num_hidden1=num_hidden1  # number of hidden nodes for 1st hidden layer
        self.num_hidden2=num_hidden2  # number of hidden nodes for 2nd hidden layer
        self.num_hidden3=num_hidden3  # number of hidden nodes for 3rd hidden layer
        self.num_hidden4=num_hidden4  # number of hidden nodes for 4th hidden layer
        self.num_hidden5=num_hidden5  # number of hidden nodes for 5th hidden layer
        self.num_hidden6=num_hidden6  # number of hidden nodes for 6th hidden layer
        self.alpha=alpha  # learning rate
        self.max_epochs=max_epochs # maximum number of epochs
        self.num_output=num_output # number of output nodes
        self._EPSILON=_EPSILON
        self.loss = [] #list to store losses per 100 epochs 
        self.trainingaccur=[] # list to store training accuracy per 100 epochs 
        self.devaccur=[]
        self.Weights_Input_to_H1=np.random.randn(self.num_hidden1, self.num_features)*(0.1)
        self.Bias_Input_to_H1=np.zeros([self.num_hidden1,1])
        self.Weights_H1_to_H2=np.random.randn(self.num_hidden2, self.num_hidden1)*(0.1)
        self.Bias_H1_to_H2=np.zeros([self.num_hidden2,1])
        self.Weights_H2_to_H3=np.random.randn(self.num_hidden3, self.num_hidden2)*(0.1)
        self.Bias_H2_to_H3=np.zeros([self.num_hidden3,1])
        self.Weights_H3_to_H4=np.random.randn(self.num_hidden4, self.num_hidden3)*(0.1)
        self.Bias_H3_to_H4=np.zeros([self.num_hidden4,1])
        self.Weights_H4_to_H5=np.random.randn(self.num_hidden5, self.num_hidden4)*(0.1)
        self.Bias_H4_to_H5=np.zeros([self.num_hidden5,1])
        self.Weights_H5_to_H6=np.random.randn(self.num_hidden6, self.num_hidden5)*(0.1)
        self.Bias_H5_to_H6=np.zeros([self.num_hidden6,1])
        self.Weights_H6_to_output=np.random.randn(self.num_output, self.num_hidden6)*(0.1)
        self.Bias_H6_to_output=np.zeros([self.num_output,1])
        self.dWeights_Input_to_H1=np.zeros([self.num_hidden1, self.num_features])
        self.dBias_Input_to_H1=np.zeros([self.num_hidden1,1])
        self.dWeights_H1_to_H2=np.zeros([self.num_hidden2, self.num_hidden1])
        self.dBias_H1_to_H2=np.zeros([self.num_hidden2,1])
        self.dWeights_H2_to_H3=np.zeros([self.num_hidden3, self.num_hidden2])
        self.dBias_H2_to_H3=np.zeros([self.num_hidden3,1])
        self.dWeights_H3_to_H4=np.zeros([self.num_hidden4, self.num_hidden3])
        self.dBias_H3_to_H4=np.zeros([self.num_hidden4,1])
        self.dWeights_H4_to_H5=np.zeros([self.num_hidden5, self.num_hidden4])
        self.dBias_H4_to_H5=np.zeros([self.num_hidden5,1])
        self.dWeights_H5_to_H6=np.zeros([self.num_hidden6, self.num_hidden5])
        self.dBias_H5_to_H6=np.zeros([self.num_hidden6,1])
        self.dWeights_H6_to_output=np.zeros([self.num_output, self.num_hidden6])
        self.dBias_H6_to_output=np.zeros([self.num_output,1])
        
        

        
    
    def relU(self,X):
        return np.maximum(X, 0)
    
    def deriv(self,X):
        return np.where(X<=0,0,1)
        
        

    
    def softmax(self,x):
        return np.exp(x - np.max(x, axis=0)) / np.sum(np.exp(x - np.max(x, axis=0)), axis=0)

    
    

    
        
    # TODO: complete implementation for forward pass
    def forward(self, X):
        self.z1=np.dot((self.Weights_Input_to_H1),(X))+self.Bias_Input_to_H1
        self.a1=self.relU(self.z1)
        self.z2=np.dot((self.Weights_H1_to_H2),(self.a1))+self.Bias_H1_to_H2
        self.a2=self.relU(self.z2)
        self.z3=np.dot((self.Weights_H2_to_H3),(self.a2))+self.Bias_H2_to_H3
        self.a3=self.relU(self.z3)
        self.z4=np.dot((self.Weights_H3_to_H4),(self.a3))+self.Bias_H3_to_H4
        self.a4=self.relU(self.z4)
        self.z5=np.dot((self.Weights_H4_to_H5),(self.a4))+self.Bias_H4_to_H5
        self.a5=self.relU(self.z5)
        self.z6=np.dot((self.Weights_H5_to_H6),(self.a5))+self.Bias_H5_to_H6
        self.a6=self.relU(self.z6)
        self.z7=np.dot((self.Weights_H6_to_output),(self.a6))+self.Bias_H6_to_output
        self.a7=self.softmax((self.z7))
        return self.a7
        
        
        
    
    # TODO: complete implementation for backpropagation
    # the following Numpy functions may be useful: np.dot, np.sum, np.tanh, numpy.ndarray.T
    def backprop(self, X, t):
            
        self.dWeights_Input_to_H1=np.zeros([self.num_hidden1, self.num_features])
        self.dBias_Input_to_H1=np.zeros([self.num_hidden1,1])
        self.dWeights_H1_to_H2=np.zeros([self.num_hidden2, self.num_hidden1])
        self.dBias_H1_to_H2=np.zeros([self.num_hidden2,1])
        self.dWeights_H2_to_H3=np.zeros([self.num_hidden3, self.num_hidden2])
        self.dBias_H2_to_H3=np.zeros([self.num_hidden3,1])
        self.dWeights_H3_to_H4=np.zeros([self.num_hidden4, self.num_hidden3])
        self.dBias_H3_to_H4=np.zeros([self.num_hidden4,1])
        self.dWeights_H4_to_H5=np.zeros([self.num_hidden5, self.num_hidden4])
        self.dBias_H4_to_H5=np.zeros([self.num_hidden5,1])
        self.dWeights_H5_to_H6=np.zeros([self.num_hidden6, self.num_hidden5])
        self.dBias_H5_to_H6=np.zeros([self.num_hidden6,1])
        self.dWeights_H6_to_output=np.zeros([self.num_output, self.num_hidden6])
        self.dBias_H6_to_output=np.zeros([self.num_output,1])
        self.dz7=(self.a7.reshape(self.num_output,-1)-t.reshape(self.num_output,-1))/((X.shape[1]))
        self.dBias_H6_to_output=np.sum(self.dz7,axis=1,keepdims=True)
        self.dWeights_H6_to_output=np.dot((self.dz7),self.a6.T)
        self.dz6=(np.dot(self.Weights_H6_to_output.T,self.dz7)) * (self.deriv(self.z6))
        self.dBias_H5_to_H6=np.sum(self.dz6,axis=1,keepdims=True)
        self.dWeights_H5_to_H6=np.dot((self.dz6),(self.a5.T))
        self.dz5=(np.dot(self.Weights_H5_to_H6.T,self.dz6)) * (self.deriv(self.z5))
        self.dBias_H4_to_H5=np.sum(self.dz5,axis=1,keepdims=True)
        self.dWeights_H4_to_H5=np.dot((self.dz5),(self.a4.T))
        self.dz4=(np.dot(self.Weights_H4_to_H5.T,self.dz5)) * (self.deriv(self.z4))
        self.dBias_H3_to_H4=np.sum(self.dz4,axis=1,keepdims=True)
        self.dWeights_H3_to_H4=np.dot((self.dz4),(self.a3.T))
        self.dz3=(np.dot(self.Weights_H3_to_H4.T,self.dz4)) * (self.deriv(self.z3))
        self.dBias_H2_to_H3=np.sum(self.dz3,axis=1,keepdims=True)
        self.dWeights_H2_to_H3=np.dot((self.dz3),(self.a2.T))
        self.dz2=(np.dot(self.Weights_H2_to_H3.T,self.dz3)) * (self.deriv(self.z2))
        self.dBias_H1_to_H2=np.sum(self.dz2,axis=1,keepdims=True)
        self.dWeights_H1_to_H2=np.dot((self.dz2),(self.a1.T))
        self.dz1=(np.dot(self.Weights_H1_to_H2.T,self.dz2)) * (self.deriv(self.z1))
        self.dBias_Input_to_H1=np.sum(self.dz1,axis=1,keepdims=True)
        self.dWeights_Input_to_H1=np.dot((self.dz1),X.T)
        
        
        
        
        
            
                
                
                
              
                        
                
      
        
        
    
    #TODO: complete implementation for fitting data, and change the existing code if needed
    def fit(self, x_train_data, y_train_data,x_dev_data,y_dev_data):
       
        
        
        for step in range(self.max_epochs):
            self.forward(x_train_data)
            self.backprop(x_train_data, y_train_data)
            self.CCloss=log_loss(np.transpose(y_train_data),np.transpose(self.a7),eps=self._EPSILON,normalize=True)
            self.trainingaccuracy=accuracy_score(np.argmax(y_train_data,axis=0),np.argmax(self.forward(x_train_data),axis=0))
            self.devaccuracy=accuracy_score(np.argmax(y_dev_data,axis=0),np.argmax(self.forward(x_dev_data),axis=0))
            self.Bias_H6_to_output=self.Bias_H6_to_output-((self.alpha)*(self.dBias_H6_to_output))
            self.Weights_H6_to_output=self.Weights_H6_to_output-((self.alpha)*(self.dWeights_H6_to_output))
            self.Bias_H5_to_H6=self.Bias_H5_to_H6-((self.alpha)*(self.dBias_H5_to_H6))
            self.Weights_H5_to_H6=self.Weights_H5_to_H6-((self.alpha)*(self.dWeights_H5_to_H6))
            self.Bias_H4_to_H5=self.Bias_H4_to_H5-((self.alpha)*(self.dBias_H4_to_H5))
            self.Weights_H4_to_H5=self.Weights_H4_to_H5-((self.alpha)*(self.dWeights_H4_to_H5))
            self.Bias_H3_to_H4=self.Bias_H3_to_H4-((self.alpha)*(self.dBias_H3_to_H4))
            self.Weights_H3_to_H4=self.Weights_H3_to_H4-((self.alpha)*(self.dWeights_H3_to_H4))
            self.Bias_H2_to_H3=self.Bias_H2_to_H3-((self.alpha)*(self.dBias_H2_to_H3))
            self.Weights_H2_to_H3=self.Weights_H2_to_H3-((self.alpha)*(self.dWeights_H2_to_H3))
            self.Bias_H1_to_H2=self.Bias_H1_to_H2-((self.alpha)*(self.dBias_H1_to_H2))
            self.Weights_H1_to_H2=self.Weights_H1_to_H2-((self.alpha)*(self.dWeights_H1_to_H2))
            self.Bias_Input_to_H1=self.Bias_Input_to_H1-((self.alpha)*(self.dBias_Input_to_H1))
            self.Weights_Input_to_H1=self.Weights_Input_to_H1-((self.alpha)*(self.dWeights_Input_to_H1))

            if step % 100 == 0:
                print(f'step: {step},  loss: {self.CCloss:3.150f}') 
                print(accuracy_score(np.argmax(y_train_data,axis=0),np.argmax(self.forward(x_train_data),axis=0)))
                print(accuracy_score(np.argmax(y_dev_data,axis=0),np.argmax(self.forward(x_dev_data),axis=0)))
                self.loss.append(self.CCloss)
                self.trainingaccur.append(self.trainingaccuracy)
                self.devaccur.append(self.devaccuracy)
                
              
            
            
    def predict(self,X,y=None):
        self.forward(X)
        if(self.num_output>1):
            y_hat=np.argmax(self.a7, axis=0)
            temp=accuracy_score(y_hat,y)
        else:
            y_hat=np.where(self.a7>0.5,1,0)
            temp=accuracy_score(y_hat,y)
        return temp,y_hat
```

## Firstly, we load the dataset through importing the sklearn library.

```python
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# import some data to play with
mnist = load_digits()
X=mnist.data
Y=mnist.target
```

```python
X.shape,Y.shape
```

![截屏2022-08-26 下午6.34.11.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-26_%25E4%25B8%258B%25E5%258D%25886.34.11.png)

- On checking its shapes, it would seem that for X the row number denotes the mth sample data, whilst there are 64 features in the data.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.11)

```

- On splitting using train_test_split, we get the following training and development and testing datasets respectively:

![截屏2022-08-26 下午6.37.02.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-26_%25E4%25B8%258B%25E5%258D%25886.37.02.png)

## On splitting, the training dataset is further scaled down using the min_max scaler from the sklearn library:

```python
import sklearn as sk
scaler=sk.preprocessing.MinMaxScaler()
```

```python
for a in range(X_train.shape[0]):
  X_train[a,:]=scaler.fit_transform(X_train[a,:].reshape(-1, 1)).flatten()
```

- One-hot encoding is then done on all training and dev true-value vectors (i.e Y):

```python
Y_train=np.array(pd.get_dummies(np.array(Y_train)))
Y_dev=np.array(pd.get_dummies(np.array(Y_dev)))
```

### The Y_train vector then becomes a matrix with 1 denoting the value each image corresponds to:

![截屏2022-08-26 下午6.40.17.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-26_%25E4%25B8%258B%25E5%258D%25886.40.17.png)

### The dataset is then further transposed:

```python
X_train=np.transpose(X_train)
X_dev=np.transpose(X_dev)
Y_train=np.transpose(Y_train)
Y_dev=np.transpose(Y_dev)
```

- Yielding the output:

![截屏2022-08-26 下午6.42.00.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-26_%25E4%25B8%258B%25E5%258D%25886.42.00.png)

### The class is then initialised:

```python
numHidden1 = 500 # number of hidden nodes
numHidden2 = 500# number of hidden nodes
numHidden3 = 500# number of hidden nodes
numHidden4 = 500# number of hidden nodes
numHidden5 = 500# number of hidden nodes
numHidden6 = 500# number of hidden nodes
num_features = X_train.shape[0]
numOutput = Y_train.shape[0]
max_epoches = 1000000
alpha = 0.01
epsilon=0.00000000001
NN = NeuralNet(num_features, numHidden1,numHidden2,numHidden3,numHidden4,numHidden5,numHidden6, alpha, max_epoches, numOutput,epsilon)
```

- where alpha is the learning rate and epsilon is there to prevent the predicted outputs =0 (as log(0) is undefined)
- The weights are randomly chosen from the standard normal distribution with sd=0.1, as using 0 as the initial weights will cause the network to become symmetrical , i.e same outputs for each training data fed into the network.
- The standard deviation= 0.1 as the network is relatively deep, hence weights too close to 0 will cause the symmetry problem, regardless of whether or not they are from the normal distribution.
- The biases are set to 0.

### The dataset is then fitted:

```python
NN.fit(X_train,Y_train,X_dev,Y_dev)
```

## Weights and biases are updated using the gradient descent update rule, i.e:

## $W_{updated} =W_{old}- \alpha * \frac{ \partial L}{ \partial W_{old}}$

## $Bias _{updated} =Bias_{old}- \alpha * \frac{ \partial L}{ \partial Bias_{old}}$

- Where $\alpha$ is the learning rate.

### The resulting test accuracy is then :

```python
accuracy_score((Y_test),np.argmax(NN.forward(X_test.T),axis=0))
```

![截屏2022-08-28 下午6.29.15.png](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/%25E6%2588%25AA%25E5%25B1%258F2022-08-28_%25E4%25B8%258B%25E5%258D%25886.29.15.png)

- 96.11% test accuracy! Rather high huh? If the network were deeper it is even possible to obtain 99.9999% accuracy and above!

# 4) Investigation on training loss, training and dev accuracy.

## Let us first investigate the training loss:

```python
import matplotlib.pyplot as plt
x_loss=range(0,len(NN.loss)*100,100)

line1=plt.plot(x_loss,NN.loss,linestyle='-',label='training loss')  

plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Untitled.png)

- Evidently the loss decreases monotonically to 0.

## Let us investigate the Training and Dev accuracies:

```python
x_training_accur=range(0,len(NN.trainingaccur)*100,100)
x_devaccur=range(0,len(NN.devaccur)*100,100)

line1=plt.plot(x_training_accur,NN.trainingaccur,linestyle='-',label='training accuracy') 
line2=plt.plot(x_devaccur,NN.devaccur,linestyle='-',label='dev accuracy')
                              
plt.title('Training and Dev Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](6%20Hidden%20layer%20Artificial%20Neural%20Network%20from%20scra%206b7dae8098034ef3bb153c6d9e41cd61/Untitled%201.png)

- Evidently the training accuracy increases as more epochs occur, and caps out at 100%.
- The dev accuracy however hits its highest at around 25000-30000 epochs, indicating that the NN isn’t learning any further beyond those epochs. The higher training accuracy also indicates overfitting on the training dataset.

# 5) Conclusion

<aside>
🌠 We managed to build a 8 layer NN that manages to classify images at a rather high accuracy. It might be better to increase the number of hidden nodes to 1000, or add more layers to make the NN more capable at learning patterns from given training data. The issue with overfitting can be remedied by stopping the training once the dev loss /accuracy is no longer decreasing/increasing.

</aside>
