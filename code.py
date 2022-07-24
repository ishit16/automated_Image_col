
from google.colab import files
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import pathlib
import tarfile
import tensorflow_datasets as tfds
from google.colab import auth
from google.colab import drive
import tensorflow_addons as tfa
import tensorflow_io as tfio
import tensorflow as tf
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
else:
    print(gpu_info)

!python - -version

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
print("Tensorflow version " + tf.__version__)

!pip install tensorflow-io tensorboardcolab tensorflow-addons & > / dev/null

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

drive.mount('/content/drive', force_remount=True)

auth.authenticate_user()

BATCH_SIZE = 32


USE_GCS = True

if not USE_GCS:
    # Run if data has been downloaded into Google Drive.
    !mkdir - p ~/tensorflow_datasets/downloads/
    !cp / content/drive/MyDrive/Dataset/places365_small/* ~/tensorflow_datasets/downloads/


if USE_GCS:
    dataset, info = tfds.load('places365_small', as_supervised=True,
                              with_info=True, data_dir='gs://tensorflow_datasets_cs229')
else:
    dataset, info = tfds.load(
        'places365_small', as_supervised=True, with_info=True)

raw_training_dataset, raw_validation_dataset, raw_test_dataset = dataset[
    'train'], dataset['validation'], dataset['test']

# Enable data argumentation.
ENABLE_DATA_ARGUMENTATION = True
# Enable batching.
ENABLE_BATCH = True
DATA_SET_NAME = 'places'


with tarfile.open('/content/drive/MyDrive/Dataset/ILSVRC2012_img_val.tar') as tar:
    tar.extractall('/tmp/imagenet/validation')

IMAGE_DIR = '/tmp/imagenet'

image_count = len(list(pathlib.Path(IMAGE_DIR).glob('*/*.JPEG')))
print(f'Total image count: {image_count}')

# Use ImageDataGenerator to stream images for ImageNet.
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.1)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.1)

raw_training_dataset = tf.data.Dataset.from_generator(
    lambda: training_datagen.flow_from_directory(
        IMAGE_DIR, IMAGE_SIZE,
        seed=229, subset='training', batch_size=BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.uint8)))
raw_validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_datagen.flow_from_directory(
        IMAGE_DIR, IMAGE_SIZE,
        seed=229, subset='validation', batch_size=BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.uint8)))

# Enable data argumentation.
ENABLE_DATA_ARGUMENTATION = True
# Disable further batching. The data has been batched.
ENABLE_BATCH = False
DATA_SET_NAME = 'image_net'

!mkdir - p ~/tensorflow_datasets/downloads/
!cp / content/drive/MyDrive/Dataset/food101/* ~/tensorflow_datasets/downloads/


dataset, info = tfds.load('food101', as_supervised=True, with_info=True)
raw_training_dataset, raw_validation_dataset = dataset['train'], dataset['validation']

# Enable data argumentation.
ENABLE_DATA_ARGUMENTATION = True
# Enable batching.
ENABLE_BATCH = True
DATA_SET_NAME = 'food'

"""### Data Argumentation and Normalization
"""


def transform_from_rgb_color_space_and_decompose(image, color_space):

    if color_space == 'lab':
        lab = tfio.experimental.color.rgb_to_lab(image)
        l = lab[..., :1] / 100.
        ab = lab[..., 1:] / 128.
        return (l, ab)
    if color_space == 'yuv':
        yuv = tf.image.rgb_to_yuv(image)
        y = yuv[..., :1]
        uv = yuv[..., 1:]
        return (y, uv)
    raise ValueError(f'Unrecognized color space: {color_space}')


def compose_and_transform_to_rgb_color_space(tensors, from_color_space):

    if from_color_space == 'lab':
        return tfio.experimental.color.lab_to_rgb(
            tf.concat([tensors[0] * 100, tensors[1] * 128], -1))
    if from_color_space == 'yuv':
        return tf.image.yuv_to_rgb(tf.concat(tensors, -1))
    raise ValueError(f'Unrecognized color space: {from_color_space}')


def transform_dataset(data_set,
                      enable_data_argumentation=False,
                      enable_batch=False,
                      color_space='lab'):

    data_set = data_set.map(lambda x, y:
                            (tf.image.resize(tf.cast(x, tf.dtypes.float32), (224, 224)), y))

    # Batch.
    if enable_batch:
        data_set = data_set.batch(BATCH_SIZE)

    # Data argumentation: flipping and rotating.
    if enable_data_argumentation:
        data_argumentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ])
        data_set = data_set.map(lambda x, y: (data_argumentation(x), y))

    # Transform into CIELab color space and decompose into input and labels.
    data_set = data_set.map(lambda x, _:
                            transform_from_rgb_color_space_and_decompose(x / 255., color_space=color_space))

    return data_set.prefetch(tf.data.AUTOTUNE)


EFFECTIVE_COLOR_SPACE = 'lab'
# EFFECTIVE_COLOR_SPACE = 'yuv'

training_dataset = transform_dataset(
    raw_training_dataset,
    enable_data_argumentation=ENABLE_DATA_ARGUMENTATION,
    enable_batch=ENABLE_BATCH,
    color_space=EFFECTIVE_COLOR_SPACE)
validation_dataset = transform_dataset(
    raw_validation_dataset,
    enable_data_argumentation=False,
    enable_batch=ENABLE_BATCH,
    color_space=EFFECTIVE_COLOR_SPACE)
if 'raw_test_dataset' in locals():
    test_dataset = transform_dataset(
        raw_test_dataset,
        enable_data_argumentation=False,
        enable_batch=ENABLE_BATCH,
        color_space=EFFECTIVE_COLOR_SPACE)

"""To verify the correctness of data pipelines and get basic sense of the data set, let's spot check a few images after processing for both training and validation set. The grayscale image (left) will be the input and the colored image (right) will be the desired output."""


plt.figure(figsize=(20, 10))
for x, y in training_dataset.take(1):
    n = min(x.shape[0], 16)
    print(f'Showing {n} samples from training set')
    for i in range(n):
        # x: input, grayscale
        plt.subplot(4, 8, i * 2 + 1)
        plt.imshow(tf.squeeze(x[i]), cmap='gray')
        plt.axis('off')
        # y: ground truth
        plt.subplot(4, 8, i * 2 + 2)
        plt.imshow(compose_and_transform_to_rgb_color_space(
            (x[i], y[i]), EFFECTIVE_COLOR_SPACE))
        plt.axis('off')


plt.figure(figsize=(20, 10))
for x, y in validation_dataset.take(1):
    n = min(x.shape[0], 16)
    print(f'Showing {n} samples from validation set')
    for i in range(n):
        # x: input, grayscale
        plt.subplot(4, 8, i * 2 + 1)
        plt.imshow(tf.squeeze(x[i]), cmap='gray')
        plt.axis('off')
        # y: ground truth
        plt.subplot(4, 8, i * 2 + 2)
        plt.imshow(compose_and_transform_to_rgb_color_space(
            (x[i], y[i]), EFFECTIVE_COLOR_SPACE))
        plt.axis('off')


def get_checkpoint_path(model_name):
    return '/content/drive/MyDrive/cs229_model_checkpoints/' + model_name + '/' + DATA_SET_NAME + '_model_{epoch:02d}_{val_loss:.6f}.hdf5'


def get_saved_model_path(model_name):
    return f'/content/drive/MyDrive/cs229_models/{model_name}/{DATA_SET_NAME}_saved_model_weights.hdf5'


"""#### Naive ConvNet

This model is being used as a baseline, which utilizes 5 Conv2D layers as feature encoder and 5 Conv2DTranspose layers as feature decoder.
"""


def build_naive_cnn_model():
    # Input layer
    input = tf.keras.layers.Input((224, 224, 1))

    # Encoders
    x = tf.keras.layers.Conv2D(64, (3, 3), 2, padding='same')(
        input)  # (224, 224) -> (112, 112)
    x = tf.keras.layers.BatchNormalization()(x)
    conv_activation_112x112 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), 2, padding='same')(
        conv_activation_112x112)  # (112, 112) -> (56, 56)
    x = tf.keras.layers.BatchNormalization()(x)
    conv_activation_56x56 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), 2, padding='same')(
        conv_activation_56x56)  # (56, 56) -> (28, 28)
    x = tf.keras.layers.BatchNormalization()(x)
    conv_activation_28x28 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), 2, padding='same')(
        conv_activation_28x28)  # (28, 28) -> (14, 14)
    x = tf.keras.layers.BatchNormalization()(x)
    conv_activation_14x14 = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), 2, padding='same')(
        conv_activation_14x14)  # (14, 14) -> (7, 7)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Decoders
    x = tf.keras.layers.Conv2DTranspose(
        512, (3, 3), 2, padding='same')(x)  # (7, 7) -> (14, 14)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.add(x, conv_activation_14x14)

    x = tf.keras.layers.Conv2DTranspose(
        256, (3, 3), 2, padding='same')(x)  # (14, 14) -> (28, 28)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.add(x, conv_activation_28x28)

    x = tf.keras.layers.Conv2DTranspose(
        128, (3, 3), 2, padding='same')(x)  # (28, 28) -> (56, 56)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.add(x, conv_activation_56x56)

    x = tf.keras.layers.Conv2DTranspose(
        64, (3, 3), 2, padding='same')(x)  # (56, 56) -> (112, 112)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.add(x, conv_activation_112x112)

    x = tf.keras.layers.Conv2DTranspose(
        32, (3, 3), 2, padding='same')(x)  # (112, 112) -> (224, 224)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.concatenate([x, input])

    output = tf.keras.layers.Conv2D(2, (1, 1), activation='tanh')(x)

    return tf.keras.models.Model(input, output)


cnn_model_naive = build_naive_cnn_model()

cnn_model_naive.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
cnn_model_naive.summary()

# Load saved model weights.
cnn_model_naive.load_weights(get_saved_model_path('cnn_model_naive'))

cnn_model_naive_checkpoint_path = get_checkpoint_path('cnn_model_naive')

assert EFFECTIVE_COLOR_SPACE == 'lab'

# Create a callback that saves the model's weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cnn_model_naive_checkpoint_path,
    verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    '/tmp/tensorboard_logs/cnn_model_naive',
    histogram_freq=1)

cnn_model_naive_history = cnn_model_naive.fit(
    training_dataset,
    validation_data=validation_dataset,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback])

"""#### ConvNet + VGG-16 Feature Extractor

This model replicates the idea proposed in [Automatic Colorization](https://tinyclouds.org/colorize/) by Ryan Dahl. Instead of utilizing only one layer of feature output, this model suggested multiple layers with different resolutions might all be useful. So it leverages the intermediate output of 4 inner layers from VGG-16 model to generate the final output. By utilizing transfer learning, this approach greatly reduces the model complexity and leads faster convergence.

This model assumes the input/output is in YUV color space.
"""


def build_feature_extractor():
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet')
    return tf.keras.models.Model(
        vgg16.input, {
            'embedding_224x224': vgg16.get_layer('block1_conv2').output,
            'embedding_112x112': vgg16.get_layer('block2_conv2').output,
            'embedding_56x56': vgg16.get_layer('block3_conv3').output,
            'embedding_28x28': vgg16.get_layer('block4_conv3').output,
        }, name='VGG16FeatureExtractor')


def build_vgg16_colorizer():
    # Input layer
    input = tf.keras.layers.Input((224, 224, 1))
    vgg16_input = tf.keras.applications.vgg16.preprocess_input(
        tf.image.grayscale_to_rgb(input) * 255.)

    vgg16 = build_feature_extractor()
    vgg16.trainable = False
    embeddings = vgg16(vgg16_input, training=False)

    # 28x28
    x = tf.keras.layers.BatchNormalization()(embeddings['embedding_28x28'])
    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 56x56
    x = tf.add(tf.keras.layers.UpSampling2D()(x),
               tf.keras.layers.BatchNormalization()(embeddings['embedding_56x56']))
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 112x112
    x = tf.add(tf.keras.layers.UpSampling2D()(x),
               tf.keras.layers.BatchNormalization()(embeddings['embedding_112x112']))
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 224x224
    x = tf.add(tf.keras.layers.UpSampling2D()(x),
               tf.keras.layers.BatchNormalization()(embeddings['embedding_224x224']))
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # 224x224
    x = tf.add(x,
               tf.keras.layers.BatchNormalization()(vgg16_input))
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        2, (3, 3), padding='same', activation='sigmoid')(x)

    output = x - 0.5

    return (tf.keras.models.Model(input, output), vgg16)


vgg16_colorizer, vgg16 = build_vgg16_colorizer()

vgg16_colorizer.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
vgg16_colorizer.summary()

# Load saved model weights.
vgg16_colorizer.load_weights(get_saved_model_path('vgg16_colorizer'))

vgg16_colorizer_checkpoint_path = get_checkpoint_path('vgg16_colorizer')

assert EFFECTIVE_COLOR_SPACE == 'yuv'

# Create a callback that saves the model's weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=vgg16_colorizer_checkpoint_path,
    save_weights_only=True,
    verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    '/tmp/tensorboard_logs/vgg16_colorizer',
    histogram_freq=1)

history = vgg16_colorizer.fit(
    training_dataset,
    validation_data=validation_dataset,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback])


def build_feature_extractor():
    # Pretrained EfficientNet B7
    efficient_net = tf.keras.applications.EfficientNetB7(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet')
    return tf.keras.models.Model(
        efficient_net.input, {
            'embedding_112x112': efficient_net.get_layer('block2a_expand_activation').output,
            'embedding_56x56': efficient_net.get_layer('block3a_expand_activation').output,
            'embedding_28x28': efficient_net.get_layer('block4a_expand_activation').output,
            'embedding_14x14': efficient_net.get_layer('block6a_expand_activation').output,
            'embedding_7x7': efficient_net.get_layer('top_activation').output,
        }, name='EfficientNetFeatureExtractor')


def build_efficient_net_colorizer():
    # Input layer
    input = tf.keras.layers.Input((224, 224, 1))
    efficient_net_input = tf.image.grayscale_to_rgb(input) * 255.

    efficient_net = build_feature_extractor()
    efficient_net.trainable = False
    embeddings = efficient_net(efficient_net_input, training=False)

    # 7x7
    x = tf.keras.layers.Conv2D(1344, (1, 1), padding='same')(
        embeddings['embedding_7x7'])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 14x14
    x = tf.add(tf.keras.layers.UpSampling2D()(
        x), embeddings['embedding_14x14'])
    x = tf.keras.layers.Conv2D(480, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 28x28
    x = tf.add(tf.keras.layers.UpSampling2D()(
        x), embeddings['embedding_28x28'])
    x = tf.keras.layers.Conv2D(288, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 56x56
    x = tf.add(tf.keras.layers.UpSampling2D()(
        x), embeddings['embedding_56x56'])
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 112x112
    x = tf.add(tf.keras.layers.UpSampling2D()(
        x), embeddings['embedding_112x112'])
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 224x224
    x = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(x), input])
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(2, (3, 3), padding='same', activation='tanh')(x)
    output = x

    return (tf.keras.models.Model(input, output), efficient_net)


efficient_net_colorizer, efficient_net = build_efficient_net_colorizer()

efficient_net_colorizer.compile(
    optimizer=tf.keras.optimizers.Adam(), loss='mse')
efficient_net_colorizer.summary()

tf.keras.utils.plot_model(efficient_net_colorizer)

# Load saved model weights.
efficient_net_colorizer.load_weights(
    get_saved_model_path('efficient_net_colorizer'))

efficient_net_colorizer_checkpoint_path = get_checkpoint_path(
    'efficient_net_colorizer')


assert EFFECTIVE_COLOR_SPACE == 'lab'

# Create a callback that saves the model's weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=efficient_net_colorizer_checkpoint_path,
    save_weights_only=True,
    verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    f'/tmp/tensorboard_logs/efficient_net_colorizer/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
    histogram_freq=1)

history = efficient_net_colorizer.fit(
    training_dataset,
    validation_data=validation_dataset,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback])


def map_ab_into_int(ab):
    return tf.cast(
        tf.minimum(tf.maximum(tf.math.round(ab * 128.) + 128., 0.), 255.),
        tf.dtypes.int32)


def map_ab_into_bucket(ab):
    ab = map_ab_into_int(ab)
    ab = ab[..., 0] * 256 + ab[..., 1]
    return ab


if False:
    assert EFFECTIVE_COLOR_SPACE == 'lab'
    # Loop through all training set for counting the probability of (a, b)
    counts = tf.zeros(65536, tf.dtypes.int64)
    i = 0
    for x, _ in raw_training_dataset.batch(BATCH_SIZE):
        x = tf.cast(x, tf.dtypes.float32) / 255.
        _, ab = transform_from_rgb_color_space_and_decompose(
            x, EFFECTIVE_COLOR_SPACE)
        # Maps into integers in [0, 65536). 2^16 buckets for possibility estimatation.
        ab = map_ab_into_bucket(ab)
        count = tf.cast(tf.math.bincount(ab), tf.int64)
        if count.shape[0] < 65536:
            count = tf.concat(
                [count, tf.zeros(65536 - count.shape[0], dtype=tf.dtypes.int64)], axis=-1)
        counts += count
        i += 1
        if i % 5000 == 0:
            print(f'Finished {i * BATCH_SIZE} examples')
    probability = tf.reshape(
        tf.cast(counts, tf.float32) /
        tf.cast(tf.reduce_sum(counts), tf.float32),
        (256, 256))
else:
    probability = np.load(
        '/content/drive/MyDrive/cs229_models/efficient_net_colorizer_v2/ab_probability_256x256.npy')


def plot_2d(distribution, name='', log_scale=False):
    '''
    Plot 2d distribution of (a, b)
    '''
    b, a = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
    plt.contourf(b, a, tf.math.log(distribution)
                 if log_scale else distribution)
    plt.axis('equal')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.colorbar()
    plt.title(name)
    plt.show()


FILTER_SHAPE = (10, 10)
GAUSSIAN_SIGMA = 5

plot_2d(probability, 'log(p(a, b))', log_scale=True)
smooth_probability = tfa.image.gaussian_filter2d(
    probability, FILTER_SHAPE, sigma=GAUSSIAN_SIGMA)
plot_2d(smooth_probability, 'smoothed log(p(a, b))', log_scale=True)

LAMBDA = 0.9

adjusted_probability = (1 - LAMBDA) * smooth_probability + \
    LAMBDA / (probability.shape[0] * probability.shape[1])
plot_2d(adjusted_probability, 'log(adjusted p(a, b))', log_scale=True)

ab_weight = 1 / adjusted_probability
ab_weight = ab_weight / tf.reduce_sum(ab_weight * adjusted_probability)
plot_2d(ab_weight, 'v(a, b)', log_scale=False)

"""###### Neural Network"""

ab_weight = tf.reshape(ab_weight, (256 * 256, 1))


def rebalanced_loss_function(y_true, y_pred):
    weight = tf.nn.embedding_lookup(ab_weight, map_ab_into_bucket(y_true))
    return tf.reduce_mean(
        weight * tf.math.squared_difference(y_true, y_pred),
        axis=(-1, -2, -3))


efficient_net_colorizer_v2, efficient_net = build_efficient_net_colorizer()
efficient_net_colorizer_v2.compile(
    optimizer=tf.keras.optimizers.Adam(), loss=rebalanced_loss_function)
efficient_net_colorizer_v2.summary()

# Load saved model weights.
efficient_net_colorizer_v2.load_weights(
    get_saved_model_path('efficient_net_colorizer_v2'))

efficient_net_colorizer_v2_checkpoint_path = get_checkpoint_path(
    'efficient_net_colorizer_v2')


assert EFFECTIVE_COLOR_SPACE == 'lab'

# Create a callback that saves the model's weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=efficient_net_colorizer_v2_checkpoint_path,
    save_weights_only=True,
    verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    f'/tmp/tensorboard_logs/efficient_net_colorizer_v2/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
    histogram_freq=1)

history = efficient_net_colorizer_v2.fit(
    training_dataset,
    validation_data=validation_dataset,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback])

# Select the model used for evaluation.
assert EFFECTIVE_COLOR_SPACE == 'lab'
model = efficient_net_colorizer_v2


evaluation_dataset = validation_dataset
evaluation_dataset_name = 'validation set'
if 'test_dataset' in locals():
    evaluation_dataset = test_dataset
    evaluation_dataset_name = 'test set'

plt.figure(figsize=(6, 16))
for x, y in evaluation_dataset.skip(random.randint(0, 128)).take(1):
    n = min(x.shape[0], 8)
    y_pred = model.predict(x)
    indicies = list(range(x.shape[0]))
    random.shuffle(indicies)
    print(f'Showing {n} samples from {evaluation_dataset_name}')
    for t in range(n):
        i = indicies[t]
        # x: input, grayscale
        plt.subplot(8, 3, t * 3 + 1)
        plt.imshow(tf.squeeze(x[i]), cmap='gray')
        plt.axis('off')
        # y_pred: prediction
        plt.subplot(8, 3, t * 3 + 2)
        plt.imshow(compose_and_transform_to_rgb_color_space(
            (x[i], y_pred[i]), EFFECTIVE_COLOR_SPACE))
        plt.axis('off')
        # y: ground truth
        plt.subplot(8, 3, t * 3 + 3)
        plt.imshow(compose_and_transform_to_rgb_color_space(
            (x[i], y[i]), EFFECTIVE_COLOR_SPACE))
        plt.axis('off')

"""#### Manually Uploaded Data Set"""


uploaded = files.upload()
filenames = list(uploaded.keys())
n = len(filenames)

plt.figure(figsize=(15, 5 * n))
for i in range(n):
    original = plt.imread(filenames[i])
    assert original.ndim == 2 or original.ndim == 3
    height, width = original.shape[0], original.shape[1]
    if original.ndim == 2:
        original = tf.reshape(original, (height, width, 1))

    # Normalize the input.
    image = tf.cast(original, tf.dtypes.float32)
    # It could be 3-channel or 1-channel image.
    if image.ndim == 3 and image.shape[2] == 3:
        image = tf.image.rgb_to_grayscale(image)
    if not (image.ndim == 3 and image.shape[2] == 1):
        raise ValueError(f'Unexpected image shape: {image.shape}')
    # Scale into [0, 1].
    image = image / 255.
    # Resize to (224, 224).
    image_224x224 = tf.image.resize(image, (224, 224))

    x = image_224x224
    y_pred = model.predict(tf.expand_dims(x, axis=0))[0]
    # x: input, grayscale
    plt.subplot(n, 3, i * 3 + 1)
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')
    # y_pred: prediction
    plt.subplot(n, 3, i * 3 + 2)
    plt.imshow(compose_and_transform_to_rgb_color_space(
        (image, tf.image.resize(y_pred, (height, width))), EFFECTIVE_COLOR_SPACE))
    plt.axis('off')
    # original: the original uploaded image
    plt.subplot(n, 3, i * 3 + 3)
    if original.shape[2] == 1:
        plt.imshow(tf.squeeze(original), cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(original)
    plt.axis('off')

"""### Compare All Models

#### Test Data Set
"""


evaluation_dataset = raw_validation_dataset
evaluation_dataset_name = 'validation set'
if 'test_dataset' in locals():
    evaluation_dataset = raw_test_dataset
    evaluation_dataset_name = 'test set'

models = [cnn_model_naive, vgg16_colorizer,
          efficient_net_colorizer, efficient_net_colorizer_v2]
color_spaces = ['lab', 'yuv', 'lab', 'lab']

rand = random.randint(0, 328000)
n = 32
m = len(models) + 2
scale = 5
plt.figure(figsize=(scale * m, scale * n))
print(
    f'Showing {n} samples from {evaluation_dataset_name}, range from {rand} to {rand + n}')
i = 0
for image, _ in evaluation_dataset.skip(rand).take(n):
    image = tf.image.resize(
        tf.cast(image, tf.dtypes.float32), (224, 224)) / 255.
    x = transform_from_rgb_color_space_and_decompose(image, 'lab')[0]
    # x: input, grayscale
    plt.subplot(n, m, i * m + 1)
    plt.imshow(tf.squeeze(x), cmap='gray')
    plt.axis('off')

    for j in range(len(models)):
        y_pred = models[j].predict(tf.expand_dims(x, axis=0))[0]
        # y_pred: prediction
        plt.subplot(n, m, i * m + 2 + j)
        plt.imshow(compose_and_transform_to_rgb_color_space(
            (x, y_pred), color_spaces[j]))
        plt.axis('off')

    # y: ground truth
    plt.subplot(n, m, i * m + m)
    plt.imshow(image)
    plt.axis('off')

    i += 1
