from skimage.util import random_noise
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import cv2
import os
import glob
from keras.layers import Input, Dense, Lambda

regularizer = keras.regularizers.l1_l2(0.01)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0,stddev=0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def build_encoder(latent_dim, shape):
    encoder_inputs = keras.Input(shape=shape)
    regularizer = keras.regularizers.l1_l2(0.01)
    x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same", 
                      kernel_regularizer=regularizer)(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", 
                      kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(72, 3, activation="relu", strides=1, padding="same",
                      kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    z = layers.Dense(latent_dim, name="z")(x)
    z_sig = layers.Dense(latent_dim, activation='softplus')(x)
    encoder = keras.Model(encoder_inputs, [z, z_sig], name="encoder")
    return encoder

def build_decoder(latent_dim, shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    regularizer = keras.regularizers.l1_l2(0.01)
    x = layers.Dense(shape[0] * shape[1] * 16, activation="relu",
                    kernel_regularizer=regularizer)(latent_inputs)
    x = layers.Reshape((shape[0]//4, shape[1]//4, 256))(x)
    x = layers.Conv2DTranspose(72, 3, activation="relu", strides=1,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(48, 3, activation="relu", strides=2,
                              kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1,
                               kernel_regularizer=regularizer, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, 
                              kernel_regularizer=regularizer, padding="same")(x)
    output = layers.Conv2DTranspose(3, 3, 
                                    activation="sigmoid", 
                                    kernel_regularizer=regularizer, 
                                    padding="same")(x)
    decoder = keras.Model(latent_inputs, output)
    return decoder

def build_transformation(latent_dim):
    model = Sequential()
    model.add(Dense(latent_dim, activation='relu', input_shape=(latent_dim,)))
    model.add(Dense(latent_dim, activation='relu'))
    model.add(Dense(latent_dim, activation='relu'))
    return model

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, shape):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = build_encoder(latent_dim, shape)
        self.transform = build_transformation(latent_dim)
        self.decoder = build_decoder(latent_dim, shape)
        self.sampling = Sampling()
        self.shape = shape

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        z = self.transform(z)
        return self.decode(z)

    @tf.function
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        return self.sampling([mean, logvar])

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def elbo_loss(self, z, mu, logvar, target):
        recons = self.decode(z)
        mse = tf.reduce_mean(tf.keras.losses.MSE(target, recons))
        mse *= self.shape[0] * self.shape[1]
        kld = -0.5 * tf.reduce_mean(1 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar))
        return mse + kld

    def train_step(self, inputs):
        noise = inputs[0][0]
        clean = inputs[0][1]
        # data: [batch * height * width * channel], noise image
        # label: [batch * height * width * channel], clean image
        with tf.GradientTape(persistent=True) as tape:
            m_n, var_n = self.encode(noise)
            z_n = self.transform(self.reparameterize(m_n, var_n))
            m_c, var_c = self.encode(clean)
            z_c = self.reparameterize(m_c, var_c)
            latent_loss = tf.reduce_mean(tf.keras.losses.MSE(z_c, z_n))
            elbo_loss = self.elbo_loss(z_n, m_n, var_n, clean)

        tran_grads = tape.gradient(latent_loss, self.transform.trainable_weights)
        self.optimizer.apply_gradients(zip(tran_grads, self.transform.trainable_weights))

        vae_grads = tape.gradient(elbo_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(vae_grads, self.trainable_weights))
        return {
            "elbo_loss": elbo_loss,
            "latent_loss": latent_loss,
        }

def train_model(model, clear_images, noise_images, epoch=1, batch_size=128):
    lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit((noise_images,clear_images), epochs=epoch, batch_size=batch_size)
    return model
