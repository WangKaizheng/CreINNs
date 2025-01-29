import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import datasets
from keras.utils import to_categorical
from test_core.load_test_data import load_svhn_test, load_cifar10
from tensorflow.keras.layers import Activation, Input, Dense, GlobalAveragePooling2D
import numpy as np
import time
import argparse
import yaml
import logging
import pickle

def resnet50(input_shape, num_classes):
    inputs = Input(input_shape)
    x = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3), classes=num_classes)(inputs)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='RES50')  
    return model

def load_model(full_path):
    snn = resnet50(input_shape=(32, 32, 3), num_classes=10)
    
    with open(full_path + '_weights', 'rb') as file:
        weights = pickle.load(file)  
        
    snn.set_weights(weights)
    opt = Adam(learning_rate=0.001)
    snn.compile(optimizer=opt)
    return snn

def single_model_evaluate(model, x_test, y_test, IFAcc):
    pred = model.predict(x_test)
    eps = 1e-12
    entropy = -np.sum(pred*np.log2(pred + eps), axis=-1)

    if IFAcc:
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_test, pred)
        acc = m.result().numpy()
        m.reset_state()
    else:
        acc = None
    return pred, acc, entropy

def ensembl_evaluate(preds, y_test, IFAcc):
    pred_ensemble = np.mean(preds, axis=0)
    
    eps = 1e-12
    
    tu = -np.sum(pred_ensemble*np.log2(pred_ensemble + eps), axis=-1)
    au = np.mean(-np.sum(preds*np.log2(preds + eps), axis=-1), axis=0)  
    eu = tu - au
   
    entropy = dict()
    entropy['TU'] = tu
    entropy['EU'] = eu
    entropy['AU'] = au

    if IFAcc:
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_test, pred_ensemble)
        acc = m.result().numpy()
        m.reset_state()  
    else:
        acc = None
    return pred_ensemble, acc, entropy

def snn_evaluation(dataset):
    seeds = [0, 66, 99, 314, 524, 803, 888, 908, 1103, 1208, 7509, 11840, 40972, 46857, 54833]
    
    ######### Get 15 Ensembles #########
    with open('ten_ensembles', 'rb') as file:
        DEs10 = pickle.load(file)
        
    ######### Load models #########
    model_list = list()
    for i in range(15):
        model_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/trainDE_100/'+str(seeds[i])
        snn = load_model(model_path)
        model_list.append(snn)

    ######## Unified dictionary for saving the results ########
    cifar = dict()
    cifar10 = dict()

    if dataset == 'CIFAR10':
        ######### Test on CIFAR10 dataset #########
        cifar = {'pred': [], 'acc': [], 'entro': [], 'label': []}
        cifar10 = {'pred': [], 'acc': [], 'entro': []}

        x_cifar, y_cifar = load_cifar10()
        
        ################################
        ######### Single Model #########
        ################################
        for i in range(15):
            model = model_list[i]
            pred, acc, entropy = single_model_evaluate(model, x_cifar, y_cifar, IFAcc=True)
            # Save the result of single model
            cifar['pred'].append(pred)
            cifar['acc'].append(acc)
            cifar['entro'].append(entropy)

        ################################
        ######### Ensembles-10 ##########
        ################################
        preds_15 = np.stack(cifar['pred'])

        for j in range(15):
            DEs10Index = DEs10[str(j)]
            preds = preds_15[DEs10Index,]
            pred_ensemble, acc, entropy = ensembl_evaluate(preds, y_cifar, IFAcc=True)
            # Save the result of ensemble
            cifar10['pred'].append(pred_ensemble)
            cifar10['acc'].append(acc)
            cifar10['entro'].append(entropy)


        cifar['label'] = y_cifar

    elif dataset == 'SVHN':
        ######### Test on SVHN dataset #########
        cifar = {'pred': [], 'entro': [],}
        cifar10 = {'pred': [], 'entro': [],}
        cifar5 = {'pred': [], 'entro': [],}

        (_, _), (x_svhn, y_svhn) = load_svhn_test()

        for i in range(15):           
            model = model_list[i]
            pred, _, entropy = single_model_evaluate(model, x_svhn, y_svhn, IFAcc=False)

            # Save the result of single model
            cifar['pred'].append(pred)
            cifar['entro'].append(entropy)

        ################################
        ######### Ensembles-3 ##########
        ################################
        preds_15 = np.stack(cifar['pred'])

        for j in range(15):
            DEs10Index = DEs10[str(j)]
            preds = preds_15[DEs10Index,]
            pred_ensemble, _, entropy = ensembl_evaluate(preds, y_svhn, IFAcc=False)
            # Save the result of ensemble
            cifar10['pred'].append(pred_ensemble)
            cifar10['entro'].append(entropy)

    else:
        print("Invalid Dataset Name. Try again...")
    return cifar, cifar10


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
    exp_num = 1
    dataset_name = config['Dataset']
    
    start_time = time.time()
    result, result10, result5 = snn_evaluation(dataset_name)
    end_time = time.time()
    print(end_time - start_time)
    
    full_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/testDE_100/ensemble10/' + dataset_name
    
    # Save test history
    with open(full_path + '_result', 'wb') as file:
        pickle.dump(result, file)
    with open(full_path + '_result10', 'wb') as file3:
        pickle.dump(result10, file3)
    with open(full_path + '_result5', 'wb') as file5:
        pickle.dump(result5, file5)

if __name__ == "__main__":
    main()


