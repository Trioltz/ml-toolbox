import os
import tensorflow as tf

# Machine learning toolbox functions inspired from Daniel Bourke's
# TensorFlow Deep Learning course.


def create_model_checkpoint(model_name: str,
                            save_path="model_experiments",
                            verbose=0,
                            save_best_only=True,
                            **kwargs):
    """
    TensorFlow callback ModelCheckpoint as a function simplifier.
    Creates a TensorFlow model checkpoint on defined path and in defined name.
    See more about functionalities from TensorFlow documentation
    tf.keras.callbacks.ModelCheckpoint:
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

    :param model_name: Name of the model as well as the checkpoint filename.
    :param save_path: Path to save files in.
    :param verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode
        1 displays messages when the callback takes an action.
    :param save_best_only: if save_best_only=True, it only saves when
        the model is considered the "best" and the latest best model
        according to the quantity monitored will not be overwritten.
        If filepath doesn't contain formatting options like {epoch}
        then filepath will be overwritten by each new better model.
    :return: None
    """
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path,
                                                                    model_name),
                                              monitor="val_loss",
                                              verbose=verbose,
                                              save_best_only=save_best_only,
                                              **kwargs)


def load_and_prep_image(filename: str,
                        img_shape=224,
                        normalize=True):
    """
    Reads in a jpeg image, turns it into a tensor and reshapes it to
    (224, 224, 3).

    :param filename: filename string of target image.
    :param img_shape: Size to resize the image to (default 224).
    :param normalize:
    :return:
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it to a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if normalize:
        # Normalize the image
        return img/255
    else:
        return img