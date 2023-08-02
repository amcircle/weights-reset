import tensorflow as tf

def make_model(im_shape, penultimate, num_classes, reg = {}, small_model = False):
    """
    Creates a convolutional neural network model using the Keras API.

    Parameters:
        im_shape (tuple): a tuple containing the dimensions of the input image (height, width, channels).
        penultimate (int): the number of neurons in the penultimate layer of the network.
        num_classes (int): the number of classes to predict.
        reg (dict, optional): a dictionary containing regularization parameters.
            Supported keys are:
            - 'dropout' (bool): whether to use dropout regularization before the last layer or not.
            - 'dropout_rate' (float): the dropout rate (between 0 and 1) to apply if 'dropout' is True.
            - 'gaussian_noise' (bool): whether to use Gaussian noise regularization or not.
            - 'gaussian_noise_stddev' (float): the standard deviation of the Gaussian noise to apply if 'gaussian_noise' is True.
            - 'gaussian_dropout' (bool): whether to use Gaussian dropout regularization or not.
            - 'gaussian_dropout_rate' (float): the Gaussian dropout rate (between 0 and 1) to apply if 'gaussian_dropout' is True.
            - 'dropout_penultimate' (bool): whether to use dropout regularization  before penultimate layer or not.
            - 'dropout_rate_penultimate' (float): the dropout rate (between 0 and 1) to apply if 'dropout_penultimate' is True.
            - 'gaussian_noise_penultimate' (bool): whether to use Gaussian noise regularization or not.
            - 'gaussian_noise_stddev_penultimate' (float): the standard deviation of the Gaussian noise to apply if 'gaussian_noise_penultimate' is True.
            - 'gaussian_dropout_penultimate' (bool): whether to use Gaussian dropout regularization or not.
            - 'gaussian_dropout_rate_penultimate' (float): the Gaussian dropout rate (between 0 and 1) to apply if 'gaussian_dropout_penultimate' is True.
            - 'l1l2_kernel' (bool): whether to use L1L2 regularization on the kernel or not.
            - 'l1l2_bias' (bool): whether to use L1L2 regularization on the bias or not.
            - 'l1l2_activity' (bool): whether to use L1L2 regularization on the activity or not.
            - 'l1l2_factor' (float): the regularization factor to apply if 'l1l2_kernel', 'l1l2_bias' or 'l1l2_activity' is True.

    Returns:
        model (tf.keras.Model): a Keras model instance.
    """
    for param in [
        'dropout', 'gaussian_noise', 'gaussian_dropout',
        'dropout_penultimate', 'gaussian_noise_penultimate', 'gaussian_dropout_penultimate',
        'l1l2_kernel', 'l1l2_bias', 'l1l2_activity',
    ]:
        reg.setdefault(param, False)
    
    input_im = tf.keras.Input(shape=im_shape)
    x = tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation=None, kernel_initializer='he_normal')(input_im)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation=None, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.ReLU()(x)
    if not small_model:
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Flatten()(x)
    if penultimate > 0:
        if reg and reg['dropout_penultimate']:
            x = tf.keras.layers.Dropout(reg['dropout_rate_penultimate'])(x)
        elif reg and reg['gaussian_noise_penultimate']:
            x = tf.keras.layers.GaussianNoise(reg['gaussian_noise_stddev_penultimate'])(x)
        elif reg and reg['gaussian_dropout_penultimate']:
            x = tf.keras.layers.GaussianDropout(reg['gaussian_dropout_rate_penultimate'])(x)
        x = tf.keras.layers.Dense(
            penultimate, 
            activation='relu', 
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_kernel'] else None,
            bias_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_bias'] else None,
            activity_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_activity'] else None
        )(x)
    if reg and reg['dropout']:
        x = tf.keras.layers.Dropout(reg['dropout_rate'])(x)
    elif reg and reg['gaussian_noise']:
        x = tf.keras.layers.GaussianNoise(reg['gaussian_noise_stddev'])(x)
    elif reg and reg['gaussian_dropout']:
        x = tf.keras.layers.GaussianDropout(reg['gaussian_dropout_rate'])(x)
    x = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_kernel'] else None,
        bias_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_bias'] else None,
        activity_regularizer=tf.keras.regularizers.L1L2(reg['l1l2_factor'],reg['l1l2_factor']) if reg and reg['l1l2_activity'] else None
    )(x)

    model = tf.keras.Model(input_im, x)

    return model