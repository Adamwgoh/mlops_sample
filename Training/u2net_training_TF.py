import os
import logging
from pathlib import Path

import mlflow
import tensorflow as tf
# only needed as sample, remove if you want to prod this
import tensorflow_datasets as tfds 

from common.general_util import get_logger
from Model.u2net_mobilenetv2_tf import U2NET_MobileNetV2_TF
from Datasets.Common.Tensorflow.augment import Augment, normalize


def load_sample_dataset(
                        buffer_size:int,
                        batch_size:int
    ):

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_images
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size)

    return (train_batches, test_batches), (dataset, info)

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def display(display_list):
  import matplotlib.pyplot as plt
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1) 
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(model:tf.keras.Model, image):
      pred_mask = model.predict(image[tf.newaxis, ...])
      display([image, create_mask(pred_mask)])

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_image, show_epoch_result:bool=True):
        super().__init__()
        self.model = model
        self.sample_image = sample_image
        self.show_epoch_result = show_epoch_result

    def on_epoch_end(self, epoch, logs=None):
        #clear_output(wait=True)
        if self.show_epoch_result:
            show_predictions(self.model, self.sample_image)
            print('\nSample Prediction after epoch {}\n'.format(epoch+1))

def run_training(
      train_batches,
      steps_per_epoch:int,
      model_savepath: str,
      epochs: int,
      checkpoint_savepath: str,
      tensorboard_logdir:str,
      validation_steps:int,
      save_period:int=5,
      mlflow_runname:str="u2net_tf_training",
      mlflow_experimentname:str="oxford_iiit_pet_sample",
      # mlflow_tracking_uri:str="sqlite:///db/mlruns.db",
      mlflow_tracking_uri:str="http://0.0.0.0:5000",
      logger=get_logger()
):
    #tracking_uri = "file://" + str(Path(tensorboard_logdir).absolute())
    #os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = 'File://./savedmodel'
    #os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = 'http://localhost:5000'
    # mlflow.mlflow.set_tracking_uri("http://0.0.0.0:5000")
    exp_found = mlflow.search_experiments(filter_string=f"name='{mlflow_experimentname}'")
    exp_id = None
    if len(exp_found) == 0:
      logger.info("No experiment found in db. Creating new MLFlow experiment")
      exp_id = mlflow.create_experiment(
                          mlflow_experimentname,
                          artifact_location=mlflow.get_artifact_uri(),
                          tags={"version": "Sample", "priority": "sample"},
                      )
    else:
      logger.info("Found existing experiment in DB. Setting it as Active")
      exp_id = exp_found[0]
      exp_id = exp_id.experiment_id

    exp = mlflow.set_experiment(exp_id)
    logger.info(f"MLFLOW Tracking Uri: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"Tracking logs are stored in: {mlflow.get_tracking_uri()}")
    logger.warning(f"Artifacts URI are stored in: {mlflow.get_artifact_uri()}")
    logger.info(f"Registred Model URI are stored in: {mlflow.get_registry_uri()}")
    
    
    model = U2NET_MobileNetV2_TF()
    for img, msk in train_batches.take(1):
        sample_input, sample_gt = img[0], msk[0]
        sample_pred = model.predict(sample_input[tf.newaxis, ...])
        logger.info(f"sample input type: {type(sample_input)}, sample pred type: {type(sample_pred)}")
    
    signature = mlflow.models.infer_signature(sample_input[tf.newaxis, ...].numpy(), sample_pred)

    save_period = 5
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_savepath, 
        verbose=1, 
        save_weights_only=True,
        save_freq= int(save_period * steps_per_epoch))


    # Track with MLFLow
    mlflow.tensorflow.autolog(every_n_iter=2)
    mlflow.end_run()
    with mlflow.start_run(run_name=mlflow_runname):
        runid = mlflow.active_run().info.run_id
        #mlflow.set_tag(runid, "sample", "1")
        #mlflow.set_tag(runid, "deletable", "1")

        logger.info(f"artifact_path = {mlflow.get_artifact_uri()}")
        # Trace with tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_logdir,
            write_graph=True,
            write_images=True,
            write_steps_per_second=False,
            update_freq='epoch'
        )


        # Run fit
        model_history = model.keras_model.fit(train_batches, epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  validation_data=test_batches,
                                  callbacks=[DisplayCallback(model, sample_input, show_epoch_result=False), tensorboard_callback, cp_callback])
        # Save final model as a mlflow model
        mlflow.tensorflow.save_model(model.keras_model, model_savepath, signature=signature)
        mlflow.end_run()

    return model

if __name__=="__main__":
    epochs         = 20
    batch_size     = 64
    val_subsplits  = 5
    buffer_size    = 1000
    model_savepath = f"./savedmodel/epoch_{epochs}_model"
    checkpoint_savepath = "./Logs/checkpoints"
    tensorboard_logpath = "./Tensorboard"
    logger = get_logger("u2net_training_TF_demo")

    (train_batches, test_batches), (dataset, info) = load_sample_dataset(buffer_size, batch_size)
    train_length = info.splits['train'].num_examples
    steps_per_epoch = train_length // batch_size

    validation_steps = info.splits['test'].num_examples//batch_size//val_subsplits
    run_training(
                train_batches,
                steps_per_epoch,
                model_savepath,
                epochs,
                checkpoint_savepath,
                tensorboard_logpath,
                validation_steps,
                mlflow_runname="u2net_tf_training_sample",
                mlflow_experimentname="refactor_u2net_mobilenetv2_TF_sample",
                logger=logger
    )