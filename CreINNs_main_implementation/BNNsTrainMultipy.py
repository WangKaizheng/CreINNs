import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD, Adam
from keras import datasets
from keras.utils import to_categorical
import numpy as np
import time
import argparse
import yaml
import logging
import pickle

from model.bnn_resnet50 import build_bnn_resnet50

####################################
def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def _augment_fn(images, labels):
    padding = 4
    image_size = 32
    target_size = image_size + padding*2
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config

def single_model_evaluate(model, x_test, y_test):
    
    pred = model.predict(x_test)

    pred_softmax = tf.nn.softmax(pred)

    m = tf.keras.metrics.CategoricalAccuracy()
    
    m.update_state(y_test, pred_softmax)
    acc = m.result().numpy()
    m.reset_state()
      
    return pred, acc
    
def main():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Accept a YAML file as a command-line argument
    parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Access hyperparameters from the loaded configuration
    seed = config['Seed']
    print('Applied Seed :', seed)
    model_type = 'BNNF'
    # model_type = 'BNNR'
    batch_size = config['BatchSize']
    # epochs = 150
    
    learning_rate = config['LearningRate']
    verbose = True
    print('Model Type: ', model_type)


    epochs = 150
    # Define the save path
    base_path = 'XXXXXXXXXXXX/train'
    full_path = base_path + model_type + '_150/' + str(seed)
    full_path_his = base_path + model_type +'_150/his/'+str(seed)
    
    # Set random seed
    # keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()

    # Define Learning Scheduler 
    def lr_scheduler(epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch >= 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch >= 120:
            lr *= 1e-2
        elif epoch >= 80:
            lr *= 1e-1
        return lr
    lr_scheduler_mod = lr_scheduler
    ###################################################################
    ##################### Prepare training dataset ####################
    ###################################################################
    
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # standard normalizing
    x_train = (x_train - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    
    val_samples = -10000
    
    x_val = x_train[val_samples:]
    y_val = y_train[val_samples:]
    
    
    x_train = x_train[:val_samples]
    y_train = y_train[:val_samples]

    
    BUFFER_SIZE = len(x_train)

    BATCH_SIZE_PER_REPLICA = batch_size
    
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    train_dataset = dataset_generator(x_train, y_train, batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(GLOBAL_BATCH_SIZE)


    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    ###################################################################
    ####################### Define INN Model ##########################
    ###################################################################
    with strategy.scope():
        model = build_bnn_resnet50(input_shape=(32, 32, 3), num_classes=10, model_type=model_type)
        # model = bnn_vgg16(input_shape=(32, 32, 3), num_classes=10, model_type=model_type)
        model.summary()
        model.compile(optimizer=Adam(learning_rate=learning_rate))
        
    # Define the metrics
    with strategy.scope():
        
        loss_tracker = keras.metrics.Mean(name="Loss")
        val_loss_tracker = keras.metrics.Mean(name="ValLoss")
     
        acc_tracker = keras.metrics.CategoricalAccuracy(name='Acc')
        val_acc_tracker = keras.metrics.CategoricalAccuracy(name='ValAcc')
        
    ###################################################################
    ####################### Define Train Step #########################
    ###################################################################
    @tf.function
    def elbo_loss(labels, logits):
        loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        loss_kl = tf.keras.losses.KLD(labels, logits)
        loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
        return loss

    def train_step(data):
    
        inputs, labels = data
        
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)

            loss = elbo_loss(labels, preds)
    
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pred_softmax = tf.nn.softmax(preds)

        loss_tracker.update_state(loss)
        acc_tracker.update_state(labels, pred_softmax)
        return loss

    ###################################################################
    ######################## Define Test Step #########################
    ###################################################################
    def test_step(data):
        inputs, labels = data
        preds = model(inputs, training=False)

        loss = elbo_loss(labels, preds)

        pred_softmax = tf.nn.softmax(preds)
          
        val_loss_tracker.update_state(loss)
        
        # Update validation accuracy
        val_acc_tracker.update_state(labels, pred_softmax)



    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
      per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
      return strategy.run(test_step, args=(dataset_inputs,))
    
    ###################################################################
    ######################## Train & Validate Loop ####################
    ###################################################################
    start = time.time()
    result_history = {'Acc': [], 'Loss': [], 'val_Acc': [], 'val_Loss': []}
    for epoch in range(epochs):
         
        # TRAIN LOOP

        for x in train_dist_dataset:
            _ = distributed_train_step(x)
        result_history['Acc'].append(acc_tracker.result().numpy())
        result_history['Loss'].append(loss_tracker.result().numpy())

        model.optimizer.learning_rate = lr_scheduler_mod(epoch)
        print(f'Epoch {epoch + 1}/{epochs}, Learning Rate: {model.optimizer.learning_rate.numpy()}')
        
        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)
        
        # Update to history per epoch
        result_history['val_Acc'].append(val_acc_tracker.result().numpy())
        result_history['val_Loss'].append(val_loss_tracker.result().numpy())
    
        template = ("Epoch {}, Loss: {:.4f}, Acc: {:.4f}" " TestLoss: {:.4f}, TestAcc: {:.4f}")
        
        # if verbose:
        print(template.format(epoch + 1, loss_tracker.result(), acc_tracker.result(), val_loss_tracker.result(), val_acc_tracker.result()))
    
        acc_tracker.reset_states()
        loss_tracker.reset_states()
        val_acc_tracker.reset_states()
        val_loss_tracker.reset_states()
    
        
    end = time.time()
    print(end-start)   
    result = result_history

    # Save trainig history
    with open(full_path_his + '_result_' + model_type, 'wb') as file:
        pickle.dump(result, file)  
        
    weights_to_save = model.get_weights() 
       
    with open(full_path + '_weights_' + model_type, 'wb') as file2:
        pickle.dump(weights_to_save, file2)
        
    pred, acc  = single_model_evaluate(model, x_test, y_test)
    
    print('acc: ', acc)

if __name__ == "__main__":
    main()
    