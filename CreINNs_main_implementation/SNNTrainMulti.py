# Import Necessary Packages
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD, Adam
from keras import datasets
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import numpy as np
import time
import pickle
import argparse
import yaml
from tensorflow.keras.layers import Activation, Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import apply_affine_transform

def resnet50(input_shape, num_classes):
    inputs = Input(input_shape)
    x = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3), classes=num_classes)(inputs)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='RES50')  
    
    return model
####################################
def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds



def _augment_fn(images, labels):
    # Define the data augmentation parameters
    width_shift_range = 0.1
    height_shift_range = 0.1
    horizontal_flip = True
    rotation_range = 20
    zoom_range = 0.1
    
    padding = 4
    image_size = 32
    target_size = image_size + padding*2
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)

    # Apply data augmentation similar to ImageDataGenerator
    # images = tf.image.random_flip_left_right(images)

    # Apply width shift and height shift
    images = apply_affine_transform(images, 
                                    tx=width_shift_range * image_size,
                                    ty=height_shift_range * image_size,
                                    fill_mode='reflect',
                                    row_axis=0,
                                    col_axis=1,
                                    channel_axis=2)

    # Apply rotation
    angle = tf.random.uniform(shape=[], minval=-rotation_range, maxval=rotation_range)
    images = apply_affine_transform(images,
                                    theta=angle,
                                    fill_mode='reflect',
                                    row_axis=0,
                                    col_axis=1,
                                    channel_axis=2)

    # Apply zoom
    zoom_factor = 1.0 + tf.random.uniform(shape=[], minval=-zoom_range, maxval=zoom_range)
    images = tf.image.central_crop(images, central_fraction=zoom_factor)

    return images, labels

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Accept a YAML file as a command-line argument
parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
args = parser.parse_args()

config = load_config(args.config_file)
seed = config['Seed']

print("Applied Seed: ", seed)

# verbose for training 
verbose = True
batch_size = 128

epochs = 150

## Please define path to save
full_path = 'train_results32SNN100/'+str(seed)
full_path_his = 'train_results32SNN100/his/'+str(seed)

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
    elif epoch >= 160:
        lr *= 1e-3
    elif epoch >= 120:
        lr *= 1e-2
    elif epoch >= 80:
        lr *= 1e-1
    return lr
lr_scheduler_mod = lr_scheduler

# Prepare training dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

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

train_dataset = dataset_generator(x_train, y_train, batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE_PER_REPLICA)

opt = Adam(learning_rate=0.001)
model = resnet50(input_shape=(32, 32, 3), num_classes=100)
model.compile(optimizer=opt)
model.summary()

loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # Cross-entropy loss
        ce_loss = loss_object(labels, predictions)
        loss = ce_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
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
model.compile(optimizer=opt, metrics=['acc'])
model.evaluate(x_test, y_test)

weigts_to_save = model.get_weights()
with open(full_path + '_weights', 'wb') as w:
    pickle.dump(weigts_to_save, w)

# Save trainig history
with open(full_path_his + '_result', 'wb') as file:
    pickle.dump(result_history, file)
    
    