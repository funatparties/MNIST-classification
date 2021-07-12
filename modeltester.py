#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: JoshM"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def demonstration():
    """Performs a sweep over a range of parameters and values to determine the
    best configuration for the neural net."""
    train_ds, test_ds = gen_mnist_fashion_ds()
    tester = ModelTester(train_ds, test_ds, data_name = 'Fashion MNIST')
    parameters = {'activation':['relu','tanh','sigmoid','elu'],
                  'hidden_layers':[[512],[512,200],[512,512],[200,200,200]],
                  'optimizer':['SGD','Adam','Adagrad','RMSprop']}
    results = tester.param_sweep(parameters, epochs=5)
    return results

def optimizer_demonstration():
    """Performs a sweep over optimiser parameters to fine tune the network."""
    train_ds, test_ds = gen_mnist_fashion_ds()
    tester = ModelTester(train_ds, test_ds, data_name = 'Fashion MNIST',
                         default = {'activation':'tanh','hidden_layers':[512,512],
                                    'optimizer':'adam'})
    parameters = {'optimizer_kwargs':[{'learning_rate':0.0005},{'learning_rate':0.001},{'learning_rate':0.002}]}
    results = tester.param_sweep(parameters, epochs=10)
    return results

def gen_mnist_ds():
    """Generates train and test Dataset objects from MNIST."""
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    return train_ds, test_ds

def gen_mnist_fashion_ds():
    """Generates train and test Dataset objects from MNIST Fashion."""
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    return train_ds, test_ds
    

class ModelTester:
    """A class for building and training conventional dense multi-layer
    perceptron (neural network) models.
    """
    def __init__(self, train_ds, test_ds, default = None, data_name = 'Data'):
        """Takes train and test datasets. Default is baseline model settings for
        parameter sweeps. Data name is simply for plot titles."""
        self._train = train_ds
        self._test = test_ds
        self._name = data_name
        self._default = {'hidden_layers':[512],'activation':'relu',
                             'optimizer':'sgd','optimizer_kwargs':{}}
        if default:
            self._default.update(default)
    
    @staticmethod
    def build_model(hidden_layers = [512], activation = 'relu', 
                 optimizer = 'SGD', optimizer_kwargs={}):
        """MLP factory builds models with a default set of parameters for doing sweeps."""
        #build layers
        model = keras.Sequential([Flatten(input_shape=(28,28))] +
                                          [Dense(h, activation=activation) for h in hidden_layers] + 
                                          [Dense(10)])
        #generate optimizer
        optimizer = tf.keras.optimizers.get(optimizer)
        config = optimizer.get_config()
        config.update(optimizer_kwargs) #apply custom settings
        optimizer = optimizer.from_config(config)
        
        model.compile(optimizer = optimizer,
                            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics = ['accuracy'])
        
        return model

    @staticmethod
    def train_model(model, train_ds, test_ds=None, epochs=5):
        """Trains the specified model on the provided data set.
        Return the history object."""
        return model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    
    @staticmethod
    def plot_sweep_results(results_dict, epochs, data_title='Data'):
        """Plots the results of a parameter sweep. Takes dict from param_sweep.
        Dashed lines indicate values for validation data."""
        num_params = len(results_dict.keys())
        fig, axs = plt.subplots(num_params, 2, sharex=True, #prepare plot grid
                                squeeze=False, tight_layout=True, dpi=200, figsize=(9,num_params*3))
        for ax in axs[-1,:]:
            ax.set_xlabel('Epoch') 
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        colours='bgmrck' #colours for plot lines
        x = range(1,epochs+1)
        for i,key in enumerate(results_dict):
            for j,val in enumerate(results_dict[key]):
                #plot loss
                axs[i,0].plot(x,results_dict[key][val]['loss'],'{}-'.format(colours[j]),label=str(key)+": "+str(val))
                axs[i,0].plot(x,results_dict[key][val]['val_loss'],'{}--'.format(colours[j]))
                axs[i,0].set_ylabel('Loss')
                axs[i,0].legend(loc='upper center')
                axs[i,0].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                
                #plot accuracy
                axs[i,1].plot(x,results_dict[key][val]['accuracy'],'{}-'.format(colours[j]),label=str(key)+": "+str(val))
                axs[i,1].plot(x,results_dict[key][val]['val_accuracy'],'{}--'.format(colours[j]))
                axs[i,1].set_ylabel("Accuracy")
                axs[i,1].legend(loc='lower center')
                axs[i,1].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        fig.suptitle('Parameter Sweep Results on '+data_title)
        plt.show()
        return

    def param_sweep(self, param_dict, epochs=5):
        """Takes dictionary of form {parameter:[values]} and trains models with
        given values for those parameters on the training data kept in 
        ModelTester object. Loss and accuracy history are reported."""
        key_results = {}
        for key,value_list in param_dict.items():
            value_results = {}
            for value in value_list:
                print(str(key)+': '+str(value))
                config = self._default.copy()
                config.update({key:value})
                model = ModelTester.model_from_params(**config)
                history = ModelTester.train_model(model, self._train, self._test, epochs=epochs)
                value_results[str(value)] = history.history
            key_results[key] = value_results
        ModelTester.plot_sweep_results(key_results, epochs, self._name)
        return key_results
    