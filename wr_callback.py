import tensorflow as tf
from operator import itemgetter
import numpy as np
import sys

class PrintEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.write(f"\rEpoch {epoch+1}, logs = {logs}")
        sys.stdout.flush()
        
    def on_train_end(self, logs=None):
        print()  # prints a newline character at the end of training

class WeightsReset(tf.keras.callbacks.Callback):
    """
    A Keras Callback class for resetting weights of specified layers during training.
    
    Parameters:
        layers (list): A list of dictionaries, where each dictionary contains the following keys:
            - 'layer' (tf.keras.layers.Layer): The layer to reset.
            - 'rand_lvl' (float): The probability of each weight being randomly reset.
            - 'weights_initializer' (callable): A function that returns the initial weights, with the same shape as the original weights.
        perform_reset (bool): Whether to actually perform the weight reset during training. Default is True.
        collect_stats (bool): Whether to collect weight statistics during training. Default is False.
        collect_weights (bool): Whether to collect the weights of the specified layers during training. Default is False.
        train_dataset (tf.data.Dataset): A TensorFlow dataset to evaluate the model on after each epoch. Default is None.
        every_epoch (int): The number of epochs between weight resets. Default is 1.
    
    Attributes:
        total_number_of_epochs (int): The total number of epochs the model will train for.
        weights_stats (list): A list of dictionaries containing the following keys:
            - 'mean' (list): A list of means of the weights of each specified layer.
            - 'max' (list): A list of maximum values of the weights of each specified layer.
            - 'min' (list): A list of minimum values of the weights of each specified layer.
            - 'variance' (list): A list of variances of the weights of each specified layer.
        weights_history (list): A list of lists, where each list contains the weights of the specified layers for one epoch.
    
    Methods:
        on_train_begin(self, logs=None): Method called at the beginning of training.
        on_epoch_end(self, epoch, logs=None): Method called at the end of each epoch during training.
    """
    def __init__(self, layers, perform_reset = True, collect_stats = False, collect_weights = False, train_dataset = None, every_epoch = 1):
        super().__init__()
        self.layers_for_reset = layers
        
        self.weights_stats = [
            {
                'mean': [],
                'max': [],
                'min': [],
                'variance': []
            } for l in layers
        ]
        self.weights_history = [[] for _ in range(len(layers))]
        
        self.collect_stats = collect_stats
        self.collect_weights = collect_weights
        self.perform_reset = perform_reset
        
        self.train_dataset = train_dataset
        
        self.every_epoch = every_epoch
        

    def on_train_begin(self, logs=None):
        self.total_number_of_epochs = self.params.get('epochs', -1)

    def on_epoch_end(self, epoch, logs=None):
        
        if self.train_dataset:
            train_metrics = self.model.evaluate(self.train_dataset, verbose=0, return_dict=True)
            for k in train_metrics:
                logs[k] = train_metrics[k]
        if (epoch + 1) % self.every_epoch == 0:
            for i, layer_with_configs in enumerate(self.layers_for_reset):
                
                layer, rand_lvl, weights_initializer = itemgetter("layer", "rand_lvl", "weights_initializer")(layer_with_configs)

                current_weights = layer.get_weights()

                if self.collect_weights:
                    self.weights_history[i].append( current_weights )

                if self.collect_stats:
                    self.weights_stats[i]['mean'].append( [np.mean(weights_group) for weights_group in current_weights] )
                    self.weights_stats[i]['max'].append( [np.amax(weights_group) for weights_group in current_weights] )
                    self.weights_stats[i]['min'].append( [np.amin(weights_group) for weights_group in current_weights] )
                    self.weights_stats[i]['variance'].append( [np.var(weights_group) for weights_group in current_weights] )

                if rand_lvl > 0 and epoch < self.total_number_of_epochs - 1 and self.perform_reset:
                    reset_mask = [tf.reshape(
                        tf.cast(
                            tf.random.categorical(
                                tf.math.log([[rand_lvl, 1 - rand_lvl]]
                                           ), tf.reduce_prod(weights_group.shape)),
                            tf.float32),
                       weights_group.shape) for weights_group in current_weights]
                    new_weights = [w * mask + weights_initializer(w.shape) * (1 - mask) for w, mask in zip(current_weights,reset_mask)]

                    layer.set_weights(new_weights)