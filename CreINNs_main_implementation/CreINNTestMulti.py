import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import datasets
from keras.utils import to_categorical
from test_core.load_test_data import load_svhn_test, load_cifar10
import numpy as np
import time
import argparse
import yaml
import logging
import pickle
from model.inn_resnet50_v3 import inn_resnet50

def compute_intersection_probability(upper_probs, lower_probs):
    alpha_num = 1.0 - np.sum(lower_probs, axis=-1, keepdims=True)
    alpha_denom = np.sum(upper_probs-lower_probs, axis=-1, keepdims=True)

    samples = alpha_num.shape[0]
    alpha = alpha_num/alpha_denom

    intersection_probs = (upper_probs - lower_probs) * alpha + 1.0 * lower_probs

    return intersection_probs

def single_model_evaluate(model, x_test, y_test):
    
    pred = model.predict([x_test, x_test])
    
    preds_lo, preds_up = pred
    
    intersection_probs = compute_intersection_probability(preds_up, preds_lo)
    
    m = tf.keras.metrics.CategoricalAccuracy()
    
    m.update_state(y_test, intersection_probs)
    acc = m.result().numpy()
    m.reset_state()
      
    return preds_lo, preds_up, intersection_probs, acc


def load_trained_model(model_path):
    learning_rate = 0.001
    opt=Adam(learning_rate=learning_rate)

    creinn = inn_resnet50(input_shape=(32, 32, 3), num_classes=10, predict_mod=True)
    
    creinn.compile(optimizer=opt)
    with open(model_path + '_weights', 'rb') as file:
        result = pickle.load(file)
    creinn.set_weights(result)
    
    return creinn

def creinn_evaluation(dataset):
    seeds = [0, 11840, 314, 46857, 54833, 7509, 888, 99, 1103, 1208, 40972, 524, 66, 803, 908]
    
    ######### Load models #########
    model_list = list()
    for i in range(15):
        model_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/trainINN/' + str(seeds[i])
        inn = load_trained_model(model_path)
        model_list.append(inn)

    ######## Unified dictionary for saving the results ########
    cifar = dict()

    if dataset == 'CIFAR10':
        ######### Test on CIFAR10 dataset #########
        cifar = {'lower_probs': [], 'upper_probs': [], 'int_probs': [], 'acc': [], 'label': []}
        
        (_, _), (x_cifar, y_cifar) = load_cifar10()
        ################################
        ######### Single Model #########
        ################################
        for i in range(15):
            model = model_list[i]
            lower_prob, upper_prob, intersec_prob, acc = single_model_evaluate(model, x_cifar, y_cifar)
            
            cifar['lower_probs'].append(lower_prob)
            cifar['upper_probs'].append(upper_prob)
            cifar['int_probs'].append(intersec_prob)
            cifar['acc'].append(acc)
            
        cifar['label'] = y_cifar

    elif dataset == 'SVHN':
        ######### Test on SVHN dataset #########
        cifar = {'lower_probs': [], 'upper_probs': [], 'int_probs': []}
    
        x_svhn, y_svhn = load_svhn_test()
        
        for i in range(15):           
            model = model_list[i]
            
            lower_prob, upper_prob, intersec_prob, _ = single_model_evaluate(model, x_svhn, y_svhn)      
            # Save the result of single model
            cifar['lower_probs'].append(lower_prob)
            cifar['upper_probs'].append(upper_prob)
            cifar['int_probs'].append(intersec_prob)

    else:
        print("Invalid Dataset Name. Try again...")
    return cifar


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


def main():
    
    # Accept a YAML file as a command-line argument
    parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Access hyperparameters from the loaded configuration
    dataset_name = config['Dataset']
    start_time = time.time()
    result = creinn_evaluation(dataset_name)
    end_time = time.time()
    print(end_time - start_time)
    
    full_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/testINN/'+ dataset_name
    
    # Save test history
    with open(full_path + '_result', 'wb') as file:
        pickle.dump(result, file)

if __name__ == "__main__":
    main()



