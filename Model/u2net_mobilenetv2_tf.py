import Model.pix2pix as pix2pix
import tensorflow as tf


class U2NET_MobileNetV2_TF(object):
    def __init__(self, **kwargs):
        self.OUTPUT_CLASSES = kwargs.get('outputclass', 3)
        self.input_shape    = kwargs.get('input_shape', [128,128,3])
        self.base_model     = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)
        self.loss           = kwargs.get('loss', tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True))
        self.optimizer      = kwargs.get('optimizer', kwargs.get('optimizer', 'adam'))
        self.metrics        = kwargs.get('matrics', ['accuracy'])

        # Use the activations of these layers
        self.layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 17x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        self.layer_names = kwargs.get('layers', self.layer_names)

        self.base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model_outputs)

        self.down_stack.trainable = False

        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

        self.keras_model = self.unet_model(self.OUTPUT_CLASSES)
        self.keras_model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)
        
    def predict(self, image):
       return self.keras_model.predict(image)

    def unet_model(self, output_channels:int):
      inputs = tf.keras.layers.Input(shape=self.input_shape)

      # Downsampling through the modelG
      skips = self.down_stack(inputs)
      x = skips[-1]
      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(self.up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

      # This is the last layer of the model
      last = tf.keras.layers.Conv2DTranspose(
          filters=output_channels, kernel_size=3, strides=2,
          padding='same')  #64x64 -> 128x128

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)
    

# use this for unit testing
if __name__=="__main__":
    assert U2NET_MobileNetV2_TF(), "Fail to compile model"