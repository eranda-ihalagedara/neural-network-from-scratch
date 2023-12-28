import numpy as np
import matplotlib.pyplot as plt
import NN.losses as losses
from .softmax import Softmax
import logging
from IPython.display import display

class Model:
    """
    A class representing a neural network model.

    """
    def __init__(self, layers, learning_rate=0.0001, loss='mean_squared_error', lr_decay=1, opt = 'sgd'):
        """
        Initializes a Model with the specified layers, learning rate, loss function, and learning rate decay.
        Parameters:
            - layers: list
                A list containing the layers of the model. Each list element should be an instance of any of the following
                    - Fully_Connected
                    - Softmax
            - learning_rate: float, optional, default: 0.0001
                The learning rate for training the model.
            - loss: str, 'mean_squared_error' or 'cross_entropy', optional, default: 'mean_squared_error'
                The loss function to be used during training.
            - lr_decay: float between 0 and 1, optional, default: 1
                The learning rate decay is exponential. In each epoch learning rate will update as:
                learning_rate = learning_rate * lr_decay.
            - opt: str, 'sgd' or 'rmsprop' or 'adam', default: 'sgd'
                The optimizer. Possible values: 
        """
        self.layers = layers
        self.learning_rate_init = learning_rate
        self.lr_decay = lr_decay
        self.loss_fn = self.get_loss_fn(loss)
        self.opt = opt

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.build()

    
    def build(self):
        """
         Build each layer in the model. Defaults to cross_entropy loss if the last layer is Softmax.
        """
        size_l = self.layers[0].size_out if self.layers[0].size_in == None else self.layers[0].size_in
        self.layers[0].build(size_l,0, self.opt)
        size_l = self.layers[0].size_out
        
        for i in range(1, len(self.layers)):
            self.layers[i].build(size_l, i, self.opt)
            size_l = self.layers[i].size_out

        # Default to softmax loss of last layer is softmax
        if isinstance(self.layers[-1], Softmax):
            self.loss_fn = self.get_loss_fn('cross_entropy')
            self.logger.info('Defaulting to cross_entropy')

    # Train model
    def train(self, x_train, y_train, batch_size = 32, epochs = 1, cv = None):
        """
        Train the model using the specified training data.

        Parameters:
        - x_train: numpy.ndarray
            The input training data in shape (n,m) where m is the number of records/samples
        - y_train: numpy.ndarray
            The target training data in shape (k,m) where m is the number of records/samples
        - batch_size: int, optional, default: 32
            The batch size for training.
        - epochs: int, optional, default: 1
            The number of training epochs.
        - cv: tuple or list : (numpy.ndarray, numpy.ndarray), optional, default: None
            Cross-validation set for validation. Should be in the order - (x_cv, y_cv)

        Returns:
        None
        """
        # Check paramters
        self.parameter_check(x_train, y_train, batch_size, epochs, cv)
        
        # Initialize metrics and plots
        m = x_train.shape[1]
        steps = np.ceil(m/batch_size)
        self.metrics_list = {}
        learning_rate = self.learning_rate_init

        fig = plt.figure()
        dsp = display(display_id=True)
        

        for epoch in range(epochs):

            for i in range(0, m, batch_size):
                x = x_train[:, i:min(i+batch_size,m)]
                a_l = self.predict(x)

                _, da_l = self.loss_fn(a_l,y_train[:, i:min(i+batch_size,m)])
                
                # Gradient values are clipped to be between -1e6 and 1e6 to avoid overflow during matrix multiplications
                da_l = np.maximum(-1e6,np.minimum(1e6, da_l))

                for layer in list(reversed(self.layers)):
                    da_l = layer.backward_pass(da_l)
                    layer.update_weights(learning_rate)

                # Progress bar
                self.print_progress(epoch, i, batch_size, steps)
 
            # Print metrics
            metrics = self.get_metrics(x_train, y_train, cv)
            self.print_metrics(metrics)

            # Plot metrics
            self.plot_metrics(fig, dsp)

            # Update learning rate
            learning_rate = learning_rate * self.lr_decay

        plt.close(fig)
        

    # Predict - forward pass through each layer
    def predict(self, x):
        """
        Perform a forward pass through each layer to make predictions.

        Parameters:
        - x: numpy.ndarray
            The input data in shape (n,m) where m is the number of records/samples

        Returns:
        - numpy.ndarray
            The model's predictions.
        """
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x


    # Set loss function
    def get_loss_fn(self, loss):
        """
        Set the loss function for the model.

        Parameters:
        - loss: 'mean_squared_error' or 'cross_entropy'
            The loss function to be used during training.

        Returns:
        - Loss function: mse or softmax_loss
        """
        if loss.lower() == 'mean_squared_error':
            return losses.mse
        elif loss.lower() == 'cross_entropy':
            return losses.softmax_loss
        else :
            raise Exception('\'' + str(loss) + '\' loss function not found!')


    # Get metrics
    def get_metrics(self, x_train, y_train, cv = None):
        """
        Calculate metrics based on the model's performance.

        Parameters:
        - x_train: numpy.ndarray
            The input training data.
        - y_train: numpy.ndarray
            The target training data.
        - cv: tuple or list, default: None
            Cross-validation set for validation.

        Returns:
        - dict
            A dictionary containing various metrics.
        """
        
        pred_train = self.predict(x_train)
        delta_train = pred_train-y_train
        if cv is not None:
            x_cv, y_cv = cv
            pred_cv = self.predict(x_cv)
            delta_cv = pred_cv - y_cv
            
        metrics = {}
        
        if self.loss_fn.__name__ == 'mse':
            mse_train = np.mean(np.sum(np.square(delta_train), axis=0,keepdims=True))
            metrics['loss'] = {'train': mse_train}
            if cv is not None:
                metrics['loss']['cv'] = np.mean(np.sum(np.square(delta_cv), axis=0,keepdims=True))

        elif self.loss_fn.__name__ == 'softmax_loss':
            # For numerical stability of log calculation, near-zero values are brought up to a small value - epsilon
            epsilon = 1e-10
            log_train = np.log(np.maximum(pred_train, epsilon))
            loss_train = -np.mean(np.sum(log_train * y_train, axis=0,keepdims=True))
            acc_train = (np.argmax(pred_train, axis=0) == np.argmax(y_train, axis=0)).mean()

            metrics['loss'] = {'train': loss_train}
            metrics['accuracy'] = {'train': acc_train}
            
            if cv is not None:
                log_cv = np.log(np.maximum(pred_cv, epsilon))
                loss_cv = -np.mean(np.sum(log_cv * y_cv, axis=0,keepdims=True))
                acc_cv = (np.argmax(pred_cv, axis=0) == np.argmax(y_cv, axis=0)).mean()

                metrics['loss']['cv'] = loss_cv
                metrics['accuracy']['cv'] = acc_cv

        else:
            raise Exception(f"Loss function '{self.loss_fn.__name__}' not found!")

        return metrics


    def parameter_check(self, x_train, y_train, batch_size, epochs, cv):
        """
        Check and validate the parameters passed for training the model.
        """
        err_msg = ''

        if (type(x_train) is not np.ndarray or type(y_train) is not np.ndarray):
            err_msg += '** `x_train` and `y_train` should be `numpy.ndarray` type'

        else:
            if x_train.shape[1] != y_train.shape[1]:
                err_msg += '** Number of records in x_train and y_train do not match!. Make sure x_train -> (n,m), y_train -> (k,m) in shape as numpy.ndarray type where `m` is the number or records in the training set\n' 
            
            if cv is not None:
                if (type(cv) is not tuple and type(cv) is not list):
                    err_msg += '** `cv` should be of type tuple or list. cv: [x_cv, y_cv] which contains cross-validation set.\n'
                else:
                    if cv[0].shape[1] != cv[1].shape[1]:
                        err_msg += '** Number of records in `cv` do not match!. Make sure cv: [x_cv, y_cv],  x_cv -> (n,m), y_cv -> (k,m) in shape where `m` is the number or records in the cross-validation set.\n' 
                    
                    if cv[0].shape[0] != x_train.shape[0]:
                        err_msg += '** Number of features in x_train and cv[0] do not match!.\n'
    

        if batch_size < 1:
            err_msg += '** `batch_size` should be an integer greater than or equal to 1'

        if epochs < 1:
            err_msg += '** `epochs` should be an integer greater than or equal to 1'

        
        if err_msg != '':
            raise Exception(f"Parameter ERROR! \n{err_msg}")
        

    def print_progress(self, epoch, i, batch_size, steps):
        """
        Print progress bar
        """
        percent = np.round(50*((i//batch_size + 1)/steps),2)
                
        if percent*2 < 100 :
            end_char = '\r'
        else:
            end_char = ' '
            
        print('epoch:', epoch+1,'='*percent.astype(int) + ' '*(50-percent.astype(int)),
                '{:.2f}'.format(percent*2),'/',100,
                end=end_char)

    def print_metrics(self, metrics):
        """
        Print metrics of each epoch
        """
        metrics_str = ''
        for key, value in metrics.items():
            metrics_str += '\t'+ key +': ' + '{:.4f}'.format(value['train']) + ' '
            self.metrics_list[key] = self.metrics_list.get(key, dict())
            for k, v in value.items():
                self.metrics_list[key][k] = self.metrics_list[key].get(k, []) + [v]
        print(metrics_str)

    def plot_metrics(self, fig, dsp):
        """
        Plot and update metrics on each epoch
        """
        plt.clf()
        n_metrics = len(self.metrics_list.keys())
        
        for idx, key in enumerate(self.metrics_list):
            step_list = np.arange(len(self.metrics_list[key]['train']))
            ax = plt.subplot(n_metrics, 1, idx+1)

            for set_name, values in self.metrics_list[key].items():
                ax.plot(step_list, np.array(values), label = set_name)
                
            ax.set_title(key, y=0.85)
            ax.grid(True)
            ax.legend()
        plt.xlabel('epoch')

        fig.canvas.draw()
        dsp.update(fig)
            