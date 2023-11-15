"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from libs import Solver
from torch.optim import SGD



class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        x_reshaped = x.reshape(x.shape[0], -1)
        out = x_reshaped.mm(w) + b
        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None

        # Compute gradients
        x_reshaped = x.reshape(x.shape[0], -1)
        dw = x_reshaped.t().mm(dout)
        db = dout.sum(dim=0)
        dx = dout.mm(w.t()).reshape_as(x)
        
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        # Apply ReLU function
        out = x.clamp(min=0)
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        x = cache

        # Compute gradient
        dx = dout * (x > 0)
        
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass (hint: cache = (fc_cache, relu_cache))
        """
        out = None
        cache = None

        # Linear forward pass
        out_fc,fc_cache = Linear.forward(x,w,b)

        # ReLU forward pass
        out,relu_cache=ReLU.forward(out_fc)

        cache = (fc_cache, relu_cache)
        
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        dx, dw, db = None, None, None

        fc_cache, relu_cache = cache

        # ReLU backward pass
        dout_fc=ReLU.backward(dout,relu_cache)

        # Linear backward pass
        dx, dw, db = Linear.backward(dout_fc,fc_cache)
        
        
        return dx, dw, db


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = None
    dx = None

    # Compute softmax scores
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()

    # Compute loss
    N = x.shape[0]
    loss = -log_probs[range(N), y].sum() / N

    # Compute gradient
    dx = probs.clone()
    dx[range(N), y] -= 1
    dx /= N

    return loss, dx


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)


        
    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        
        # Forward pass
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        scores = None

        # Compute the first hidden layer
        h1 = Linear.forward(X, W1, b1)[0]
        h1_relu = ReLU.forward(h1)[0]

        # Compute the scores for each class
        scores = Linear.forward(h1_relu, W2, b2)[0]

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        # Compute data loss using softmax
        loss, dsoftmax= softmax_loss(scores, y)

        # Add L2 regularization to the loss
        loss += 1 * self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

        
        grads['W2'] = torch.matmul(h1_relu.t(), dsoftmax) + 2*self.reg * W2
        grads['b2'] = torch.sum(dsoftmax, dim=0)

        dh1_relu = torch.matmul(dsoftmax, W2.t())
        dh1 = (h1 > 0).type(h1.dtype) * dh1_relu

        grads['W1'] = torch.matmul(X.t(), dh1) + 2*self.reg * W1
        grads['b1'] = torch.sum(dh1, dim=0)
            
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * torch.randn(layers_dims[i], layers_dims[i+1], dtype=dtype, device=device)
            self.params['b' + str(i+1)] = torch.zeros(layers_dims[i+1], dtype=dtype, device=device)
                          
        
        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode

        # Forward pass
        out = X
        cache = {}
        cache_dropout={}
        for i in range(1, self.num_layers + 1):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            if i == self.num_layers:
                out, cache[i] = Linear.forward(out, W, b)
            else:
                out, cache[i] = Linear_ReLU.forward(out, W, b)
                if self.use_dropout:
                    out, cache_dropout[i] = Dropout.forward(out, self.dropout_param)
                    

        scores = out

                

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # Backward pass
        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers, 0, -1):
            W = self.params['W' + str(i)]
            loss += 1 * self.reg * torch.sum(W * W)
            if i == self.num_layers:
                dout, grads['W' + str(i)], grads['b' + str(i)] = Linear.backward(dout, cache[i])
            else:
                if self.use_dropout:
                    dout = Dropout.backward(dout, cache_dropout[i])
                dout, grads['W' + str(i)], grads['b' + str(i)] = Linear_ReLU.backward(dout, cache[i])
            grads['W' + str(i)] += 2*self.reg * W

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
  model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    
  


  # Initialize a Solver instance
  solver = Solver(model, data_dict,
                update_rule=sgd,
                optim_config={
                  'learning_rate': 5*1e-2,
                },
                num_epochs=10, batch_size=100,device=device,print_every=1000)
  solver.train()

  # Train the model
  solver.train()

  return solver


def get_three_layer_network_params():
    
    weight_scale = 2   # Experiment with this!
    learning_rate = 1e-2  # Experiment with this!
    return weight_scale, learning_rate


def get_five_layer_network_params():
    
    learning_rate = 1e-3  # Experiment with this!
    weight_scale = 1   # Experiment with this!
  
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    
    # Momentum update
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    # RMSProp update
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw**2
    next_w = w - config['learning_rate'] * dw / (torch.sqrt(config['cache']) + config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    # Adam update
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw**2

    m_hat = config['m'] / (1 - config['beta1']**config['t'])
    v_hat = config['v'] / (1 - config['beta2']**config['t'])

    next_w = w - config['learning_rate'] * m_hat / (torch.sqrt(v_hat) + config['epsilon'])

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            mask = (torch.rand_like(x) >= p) / (1 - p)
            out = x * mask
          
        elif mode == 'test':
            out = x
        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout
        return dx

