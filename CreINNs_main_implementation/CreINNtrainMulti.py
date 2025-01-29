import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD, Adam
from keras import datasets
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import pickle

import argparse
import yaml

from model.inn_resnet50_v3 import inn_resnet50

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config

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
    
    pred = model.predict([x_test, x_test])
    
    preds_lo, preds_up = pred
    
    m = tf.keras.metrics.CategoricalAccuracy()
    
    m.update_state(y_test, preds_lo)
    acc_L = m.result().numpy()
    m.reset_state()
    
    m.update_state(y_test, preds_up)
    acc_U = m.result().numpy()
    m.reset_state()
      
    return pred, acc_L, acc_U
    


# Accept a YAML file as a command-line argument
parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
args = parser.parse_args()

config = load_config(args.config_file)
    
batch_size = 128

seed = config['Seed']
learning_rate = 0.001

# Please define your path to save the model and training file!!!!

epochs = 150
full_path = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/trainINN/'+str(seed)
full_path_his = '/staging/leuven/stg_00155/CreINNResults/RES50_BACKBONE/trainINN/his/'+str(seed)

# Set random seed if you need
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
    elif epoch >= 160:
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

BATCH_SIZE_PER_REPLICA = batch_size
train_dataset = dataset_generator(x_train, y_train, batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE_PER_REPLICA)

###################################################################
####################### Define INN Model ##########################
###################################################################

# model = inn_resnet(input_shape=(32, 32, 3), num_classes=10)
# model = inn_vgg16(input_shape=(32, 32, 3), num_classes=10, predict_mod=False)
model = inn_resnet50(input_shape=(32, 32, 3), num_classes=10, predict_mod=False)
model.summary()
model.compile(optimizer=Adam(learning_rate=learning_rate))

loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model([images, images], training=True)
        # Cross-entropy loss
        ce_loss = loss_object(labels, predictions)
        loss = ce_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model([images, images], training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)

start = time.time()

result_history = {'Acc': [], 'Loss': [], 'val_Acc': [], 'val_Loss': []}
curr_epoch = 0
for e in range(int(curr_epoch), epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels)

    for images, labels in test_dataset:
        test_step(images, labels)
        
    model.optimizer.learning_rate = lr_scheduler_mod(e)
    
    print(f'Epoch {e + 1}/{epochs}, Learning Rate: {model.optimizer.learning_rate.numpy()}')

    template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
    print (template.format(e+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))
    
    result_history['Acc'].append(train_accuracy.result())
    result_history['Loss'].append(train_loss.result())
    result_history['val_Acc'].append(test_accuracy.result())
    result_history['val_Loss'].append(test_loss.result())
  
end_time = time.time()
print(end_time-start)
model.compile(optimizer=Adam(learning_rate=learning_rate), metrics=['acc'], loss='categorical_crossentropy')
model.evaluate([x_test, x_test], y_test)

weigts_to_save = model.get_weights()
with open(full_path + '_weights', 'wb') as w:
    pickle.dump(weigts_to_save, w)

# Save trainig history
with open(full_path_his + '_result', 'wb') as file:
    pickle.dump(result_history, file)

