import time
import sys, os
import numpy as np
from typing import List, Callable, NewType, Optional


def relu(Z):

  """Applies relu function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying relu function
  """
  
  return np.maximum(Z, 0)



def softmax(self, X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.sum(np.exp(X), axis=0)


def softmax_stable(self, xs: np.ndarray) -> np.ndarray:
    xs = xs - np.max(xs, axis=0, keepdims=True)
    xs_exp = np.exp(xs)
    return xs_exp / xs_exp.sum(axis=0, keepdims=True)


def relu_prime(Z):
  
  """Applies differentiation of relu function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying diff of relu function
  """

  return (Z>0).astype(Z.dtype)

def sigmoid(Z):

  """Applies sigmoid function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying sigmoid function
  """    
  
  # return 1/(1+np.power(np.e, -Z))
  return 1/(1 + np.exp(-Z) )

def sigmoid_prime(Z):

  """Applies differentiation of sigmoid function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying diff of sigmoid function
  """
  
  # return Z * (1-Z)
  return (1 - sigmoid(Z)) * sigmoid(Z)

def leaky_relu(Z, alpha=0.01):

  """Applies leaky relu function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value
    alpha: float
      Negative slope coefficient

    Returns
    -------
    A: same shape as input
      Value after applying leaky relu function
  """   

  return np.where(Z > 0, Z, Z * alpha)

def leaky_relu_prime(Z, alpha=0.01):

  """Applies differentiation of leaky relu function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value
    alpha: float
      Negative slope coefficient

    Returns
    -------
    A: same shape as input
      Value after applying diff of leaky relu function
  """

  dz = np.ones_like(Z)
  dz[Z < 0] = alpha
  return dz

def tanh(Z):

  """Applies tanh function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying tanh function
  """   

  return np.tanh(Z)

def tanh_prime(Z):
  
  """Applies differentiation of tanh function to an array/value

    Arguments
    ---------
    Z: float/int/array_like
      Original Value

    Returns
    -------
    A: same shape as input
      Value after applying diff of tanh function
  """

  return 1-(tanh(Z)**2)

def linear(Z):
  return Z

def linear_prime(Z):
  return 1.

# def initialize_layer_weights(units, inputs, random_state=0):

#   """Initializes random weights and bias for a layer l

#     Arguments
#     ---------
#     inputs: int
#       Number of neurons in previous layer (l-1)
#     units: int
#       Number of neurons in current layer (l)
#     random_state: int
#       Random seed

#     Returns
#     -------
#     dict
#       Contains the randomly initialized weights and bias arrays

#       The keys for weights and bias arrays in the dict are 'W1', 'b1', 'W2' and 'b2'
#   """

#   np.random.seed(random_state)

#   w = np.random.randn(inputs, units) * np.sqrt(2/inputs)
#   b = np.random.randn(1, units) * np.sqrt(2/inputs)

#   return (w, b)


def get_activation_fcn(name: str) -> set:
    activations = {'relu': (relu, relu_prime),
                   'sigmoid': (sigmoid, sigmoid_prime),
                   'leaky_relu': (leaky_relu, leaky_relu_prime),
                   'tanh': (tanh, tanh_prime),
                   'linear': (linear, linear_prime)
                   }

    func = activations[name.lower()]
    assert len(func) == 2

    return func





class Dense(object):
  
  '''
  Returns a dense layer with randomly initialized weights and bias

    Arguments
    ---------
    input_dim: int
      Number of neurons in previous layer.
    units: int
      Number of neurons in the layer.
    activation: str
      Activation function to use. 'relu', 'leaky_relu', 'tanh', 'linear' or 'sigmoid'

    Returns
    -------
    Dense layer
      An instance of the Dense layer initialized with random params.
  '''


  def __init__(self,
               units: int,
               inputs: int,
               activation: Optional[str] = 'relu',
               learning_rate: Optional[float] = 0.1,
               init_method: Optional[str] = 'uniform',
               name: Optional[str] = 'Dense_' + str( int(time.time()) ),
               random_state: Optional[int] = None):

    self.units       = units
    self.inputs      = inputs
    self.seed        = random_state
    self.name        = name
    self.init_method = init_method
    self.activation  = activation
    
    self.W, self.b = self.init_weights()
    self.activ, self.activ_prime = get_activation_fcn(activation)

    self.Z = None
    self.A = None
    self.dz = None
    self.da = None
    self.dw = None
    self.db = None



  def init_weights(self) -> set:
    np.random.seed(self.seed)

    if self.init_method == 'uniform':
      return ( np.random.uniform(-1.e-4, 1.e-4, size=(self.units, self.inputs) ),
               np.random.uniform(-1.e-4, 1.e-4, size=(self.units, 1))
             )
    else:   
      # N(mu, sigma) normal distribution
      mu = 0.
      sigma = 1.
      return  ( (mu + sigma*np.random.randn(self.units, self.inputs)) * np.sqrt(2./self.inputs),
                (mu + sigma*np.random.randn(self.units, 1)) * np.sqrt(2./self.inputs)
              )
    # End - if
  # End - def
# End - Dense


class Network(object):
    '''

    '''
    def __init__(self,
                 layer_units: List[int],
                 layer_activation: Optional[str] = 'relu',
                 output_activation: Optional[str] = 'sigmoid',
                 init_method: Optional[str] = 'norm',
                 random_state: Optional[int] = None):
        
        self.layer_units       = layer_units
        self.layer_activation  = layer_activation
        self.output_activation = output_activation
        self.init_method       = init_method
        self.seed              = random_state

        self.inputs = None
        self.out    = None
        
        self.layers = []
        for l in range(len(self.layer_units)-1 ):
            self.layers.append( Dense(units=self.layer_units[l],
                                inputs=self.layer_units[l-1],
                                activation=self.layer_activation,
                                init_method= self.init_method,
                                random_state=self.seed)
                        )

        self.layers.append( Dense(units=self.layer_units[-1],
                                inputs=self.layer_units[-2],
                                activation=self.output_activation,
                                init_method= self.init_method,
                                random_state=self.seed)
                           )

        
        
    def _forward_prop(self, X: np.ndarray) -> np.ndarray:
        A_prev = X.copy()

        # Hidden layers
        for i in range(1, len(self.layer_units) - 1):
            self.layers[i].Z = np.dot(self.layers[i].W, A_prev) + self.layers[i].b  # Z = W.X + b
            A_prev           = self.layers[i].activ( self.layers[i].Z )             # A = f(Z) where f = activation function
            self.layers[i].A = A_prev

        # output layer, with default activation = sigmoid
        self.layers[-1].Z = np.dot(self.layers[-1].W, A_prev) + self.layers[-1].b   # Z = W.X + b
        out = self.layers[-1].activ(self.layers[-1].Z)                              # A = f(Z) where f = activation function
        self.layers[-1].A = out

        self.out = out
        return out

    
    def forward_prop(self, X: np.ndarray, model: list) -> list:
      
      """Performs forward propagation and calculates output value

        Arguments
        ---------
        X: array_like
          Data
        model: list
          List containing the layers

        Returns
        -------
        Model: list
          List containing layers with updated 'Z' and 'A'
      """

      for i in range(len(model)):

        if i==0:
          X_prev = X.copy()
        else:
          X_prev = model[i-1].A

        model[i].Z = np.dot(model[i].W, X_prev) + model[i].b
        model[i].A = model[i].activ(model[i].Z)
        
      return model




    def calculate_loss(self, y, model):

      """Calculate the entropy loss

        Arguments
        --------- 
        y: array-like
          True lables
        model: list
          List containing the layers

        Returns
        -------
        loss: float
          Entropy loss
      """

      m = y.shape[0]
      A = model[-1].A

      return np.squeeze(-(1./m)*np.sum(np.multiply(y, np.log(A))+np.multiply(np.log(1-A), 1-y)))     # Squeeze will convert [[cost]] to 'cost' float variable

    def backward_prop(self, X, y, model):

      """Performs forward propagation

        Arguments
        ---------
        X: array_like
          Data
        y: array_like
          True labels
        model: list
          List containing the layers

        Returns
        -------
        model: list
          List containing the layers with calculated 'dw' and 'db'
      """

      m = X.shape[0]

      for i in range(len(model)-1, -1, -1):

        if i==len(model)-1:
          model[i].dz = model[-1].A - y
          model[i].dw = 1./m * np.dot(model[i-1].A.T, model[i].dz)
          model[i].db = 1./m * np.sum(model[i].dz, axis=0, keepdims=True)

          model[i-1].da = np.dot(model[i].dz, model[i].W.T)

        else:

          # model[i].dz = np.multiply(np.int64(model[i].A>0), model[i].da) * get_derivative_activation_function(model[i].activation)(model[i].Z)
          model[i].dz = np.multiply(np.int64(model[i].A>0), model[i].da) * model[i].activ_prime(model[i].Z)

          if i!=0:
            model[i].dw = 1./m * np.dot(model[i-1].A.T, model[i].dz)
          else:
            model[i].dw = 1./m * np.dot(X.T, model[i].dz)
          model[i].db = 1./m * np.sum(model[i].dz, axis=0, keepdims=True)
          if i!=0:
            model[i-1].da = np.dot(model[i].dz, model[i].W.T)

      return model

    def update_weights(self, model, learning_rate=0.01):

      """Updates weights of the layers

        Arguments
        ---------
        model: list
          List containing the layers
        learning_rate: int, float
          Learning rate for the weight update

        Returns
        -------
        model: list
          List containing the layers
      """

      for i in range(len(model)):
        model[i].W -= learning_rate*model[i].dw
        model[i].b -= learning_rate*model[i].db
        
      return model

    def predict(self, X, y, model):
        
      """Using the learned parameters, predicts a class for each example in X
      
      Arguments
      ---------
      X: array_like
        Data
      y: array_like
        True Labels
      model: list
        List containing the layers
      
      Returns
      -------
      predictions: array_like
        Vector of predictions of our model
      """
      
      model1 = forward_prop(X, model.copy())
      predictions = np.where(model1[-1].A > 0.5, 1, 0)
      
      return predictions      