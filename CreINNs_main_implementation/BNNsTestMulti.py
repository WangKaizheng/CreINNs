import pickle
import numpy as np
import tensorflow as tf
import argparse
import yaml
from keras.optimizers import Adam
from model.bnn_resnet50 import build_bnn_resnet50
from test_core.load_test_data import load_svhn_test, load_cifar10
import time
from tensorflow import keras

def load_model(full_path, model_type):
    model = build_bnn_resnet50(input_shape=(32, 32, 3), num_classes=10, model_type=model_type)
    opt=Adam(learning_rate=0.001)
    model.compile(optimizer=opt)
    with open(full_path + '_weights_' + model_type, 'rb') as file:
        result = pickle.load(file)
    model.set_weights(result)
    return model
    
def single_model_evaluate(model, x_test, y_test, IFAcc):
    
    pred = model.predict(x_test)
    
    pred_softmax = tf.nn.softmax(pred)
    
    eps = 1e-12
    
    entropy = -np.sum(pred*np.log2(pred_softmax + eps), axis=-1)

    if IFAcc:
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_test, pred_softmax)
        acc = m.result().numpy()
        m.reset_state()
    else:
        acc = None
    return pred_softmax, acc, entropy

def ensembl_evaluate(preds, y_test, IFAcc):
    pred_ensemble = np.mean(preds, axis=0)
    
    # print(preds.shape)
    
    eps = 1e-12
    
    tu = -np.sum(pred_ensemble*np.log2(pred_ensemble + eps), axis=-1)
    # print(tu.shape)
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

def bnn_evaluation(dataset, model_type):
    
    seeds = [0, 66, 99, 314, 524, 803, 888, 908, 1103, 1208, 7509, 11840, 40972, 46857, 54833]
    
    num = len(seeds)      
    ######### Load models #########
    model_list = list()
    for i in range(num):
        model_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/train' + model_type + '_150/' + str(seeds[i])
        bnn = load_model(model_path, model_type)
        model_list.append(bnn)

    ######## Unified dictionary for saving the results ########
    cifar10 = dict()
    if dataset == 'CIFAR10':
        ######### Test on CIFAR10 dataset #########
        # cifar = {'pred': [], 'acc': [], 'entro': [], 'label': []}
        cifar10 = {'pred': [], 'acc': [], 'entro': [], 'label': []}
        
        (_, _), (x_cifar, y_cifar) = load_cifar10()

        ################################
        ######### Ensembles-5 ##########
        ################################
        for j in range(num):
            model = model_list[j]
            pred_softmax5 = []
            for k in range(10):
                pred_softmax, _, _ = single_model_evaluate(model, x_cifar, y_cifar, IFAcc=True)
                pred_softmax5.append(pred_softmax)
            preds = np.stack(pred_softmax5)
            
            pred_ensemble, acc, entropy = ensembl_evaluate(preds, y_cifar, IFAcc=True)
            # Save the result of ensemble
            cifar10['pred'].append(preds)
            cifar10['acc'].append(acc)
            cifar10['entro'].append(entropy)

        # cifar['label'] = y_cifar
        cifar10['label'] = y_cifar

    elif dataset == 'SVHN':
        ######### Test on SVHN dataset #########
        cifar10 = {'pred': [], 'entro': [],}

        x_svhn, y_svhn = load_svhn_test()

        ################################
        ######### Ensembles-5 ##########
        ################################
        for j in range(num):
            model = model_list[j]
            pred_softmax5 = []
            for k in range(10):
                pred_softmax, _, _ = single_model_evaluate(model, x_svhn, y_svhn, IFAcc=False)
                pred_softmax5.append(pred_softmax)
            preds = np.stack(pred_softmax5)
            
            pred_ensemble, _, entropy = ensembl_evaluate(preds, y_svhn, IFAcc=False)
            # Save the result of ensemble
            cifar10['pred'].append(preds)
            cifar10['entro'].append(entropy)
            
    else:
        print("Invalid Dataset Name. Try again...")
    return cifar10


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
    # exp_num = config['ExpNum']
    exp_num = 1
    
    dataset_name = config['Dataset']
    delta = config['Delta']
    
    # model_type = 'BNNF'
    model_type = 'BNNR'
    
    start_time = time.time()
    result = bnn_evaluation(dataset_name, model_type)
    end_time = time.time()
    print(end_time - start_time)
    
    full_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/test' + model_type + '_150/' + dataset_name

    
    # Save test history
    with open(full_path + '_result10', 'wb') as file:
        pickle.dump(result, file)


if __name__ == "__main__":
    main()
