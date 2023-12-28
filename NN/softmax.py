import NN.activations as act
import NN.optimizers as optimizers
import numpy as np

class Softmax:
    """
    A class representing a softmax neural network layer.
    
    """

    def __init__(self, size = 1, input_size = None):
        """
        Initializes a Fully_Connected layer with the specified size, activation function, and input size.
        
        Parameters:
        - size: int, default: 1
            The number of neurons in the layer.
        - input_size: int, optional, default: None
            The size of the input to the layer.

        """
        self.size_out = size
        self.size_in = input_size
        self.g = act.softmax

        
    def build(self, size_in, layer_id, opt):
        """
        Set input size to the layer and initialize the weight matrix and bias vector when building the model.

        Parameters:
        - size_in: int
            The size of the input to the layer
        - layer_id: int
            The id of the layer
        - opt: str
            The optimizer

        Returns:
        None
        
        """
        self.layer_id=layer_id
        self.size_in = size_in
        self.w = np.random.rand(self.size_out,self.size_in)-0.5
        self.b = np.zeros([self.size_out,1])
        self.opt = optimizers.get_optimizer(opt, self.w.shape, self.b.shape)

    
    def forward_pass(self, a_l_munus_1):
        """
         Perform a forward pass through the layer.
        
        Parameters:
        - a_l_munus_1: numpy.ndarray
            The input to the layer from the layer before.

        Returns:
        - numpy.ndarray
            The output of the layer after applying the activation function to the transformed input with weights and bias.
            
        """
        self.a_l_munus_1 = a_l_munus_1
        self.z = np.matmul(self.w, a_l_munus_1) + self.b
        return self.g(self.z)

    
    def backward_pass(self, dz):
        """
        Perform a backward pass through the layer.

        Parameters:
        - dz: numpy.ndarray
            The gradient of the loss with respect to the layer's `z` values.

        Returns:
        - numpy.ndarray
            The gradient of the loss with respect to the layer's input (da_l_minus_1).
            
        """
        m = dz.shape[1]
        self.dz = dz 
        self.dw = self.dz @ self.a_l_munus_1.T /m
        self.db = np.sum(self.dz, axis=1, keepdims=True)/m

        return self.w.T @ self.dz # Return da_l_munus_1
    
            
    def update_weights(self, learning_rate, grad_clip = 10):
        """
        Update the weights of the layer using gradient descent with optional gradient clipping.

        Parameters:
        - learning_rate: float between 0 and 1
            The learning rate for the gradient descent.
        - grad_clip: float, default: 1
            The threshold value for gradient clipping.

        Returns:
        None
        
        """
        dw_opt = self.opt.get_dw_opt(self.dw)
        db_opt = self.opt.get_db_opt(self.db)

        self.w -= learning_rate * np.maximum(-grad_clip,np.minimum(grad_clip, dw_opt))
        self.b -= learning_rate * np.maximum(-grad_clip,np.minimum(grad_clip, db_opt))

        # Check if nan in weights
        if np.isnan(self.w).sum() == 1:
            print('Layer:', self.layer_id,'nan in W')
            print('dw:', self.dw)
        if np.isnan(self.b).sum() == 1:
            print('Layer:', self.layer_id,'nan in b')
            print('db:', self.db)


