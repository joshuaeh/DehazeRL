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
import torch
import torchvision
import imageio
from tensorflow.keras.layers import Input, Dense, Lambda
from model import VAE

cwd = ''

'''
PREPROCESSING: LOAD DATASET
'''
def load_celeb_images(file_path):
    raw_image_dataset = tf.data.TFRecordDataset(file_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    images = []
    for image_features in parsed_image_dataset:
        image_raw = image_features['data'].numpy()
        shape = image_features['shape'].numpy()
        img = tf.io.decode_raw(image_raw, tf.uint8)
        img = tf.reshape(img, shape).numpy()
        images.append(img)
    return images

def read_images(files):
    data = []
    for f1 in files:
        img = []
        img = cv2.imread(f1)
        data.append(img)
    return data

def crop_square(imgs, length = 3000):
    data = []
    for img in imgs:
        img1, img2 = img[:length, -length:], img[:length, :length]
        data.append(cv2.resize(img1, (length,length)))
        data.append(cv2.resize(img2, (length,length)))
    return data

def recover_square(img1, img2, shape, length=3000):
    h, w = shape[0], shape[1]
    if h > w:
        ol = np.array(np.mean([img1[:w-h+length, :], img2[h-w-length:, :]], axis=0), dtype='uint8')
        img = np.concatenate([img2[:h-length, :], ol, img1[length-h:,:]], axis=0)
    else:
        ol = np.array(np.mean([img1[:, :h-w+length], img2[:, w-h-length:]], axis=0), dtype='uint8')
        print(ol.shape, img2.shape, img1.shape)
        img = np.concatenate([img2[:, :w-length], ol, img1[:, length-w:]], axis=1)
    return img

def sidd_test_data(path, key, batch):
    import scipy.io
    mat = scipy.io.loadmat(path)
    tmp = mat.get(key)
    images = []
    for i in range(batch*10,batch*10+10):
        for j in range(32):
            images.append(cv2.resize(tmp[i][j],(248,248)))
    return images

def imshow(img, rgb=True):
    if rgb:
        cv2_imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        cv2_imshow(img)

'''
PREPROCESSING: GENERATE TRAINGING AND VALIDATION DATA
'''
def gen_noise(images):
    noise = []
    for img in images:
        noise_img = random_noise(np.copy(img), mode='gaussian', var=0.02)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        noise.append(noise_img)
    return noise

def cvt_bgr_yuv(images):
    ys, us, vs = [],[],[]
    for img in images:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        ys.append(np.expand_dims(y, axis=2))
        us.append(np.expand_dims(u, axis=2))
        vs.append(np.expand_dims(v, axis=2))
    return np.array(ys, dtype=np.uint8), np.array(us, dtype=np.uint8), np.array(vs, dtype=np.uint8)

def cvt_yuv_bgr(y, u, v):
    images = []
    for i in range(len(y)):
        yuv = np.zeros((y.shape[1], y.shape[2], 3), dtype=np.uint8)
        yuv[:,:,0] = y[i,:,:,0]
        yuv[:,:,1] = u[i,:,:,0]
        yuv[:,:,2] = v[i,:,:,0]
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        images.append(img)
    return np.array(images, dtype=np.uint8)

def divide_img(img, block_size=18, num_block=18, overlap=4):
    height = len(img)
    width = len(img[0])
    if not (block_size*num_block - (num_block - 1)*overlap == height):
        raise ValueError('Block size mismatch', block_size*num_block - (num_block - 1)*overlap, height)
    size = block_size - overlap
    blocks = np.array([img[i:i+block_size, j:j+block_size] 
                      for j in range(0,width - overlap,size) 
                      for i in range(0,height - overlap,size)])
    return blocks

def merge_img(blocks, width=256, height=256, block_size=18, overlap=4):
    num_block_per_row = (width - overlap)//(block_size - overlap)
    num_block_per_col = (height - overlap)//(block_size - overlap)
    def get_row_block(row):
        row_block = blocks[row]
        for j in range(1, num_block_per_row):
            cur_row_block = row_block[:, :len(row_block[0]) - overlap]
            block1 = blocks[row+j*num_block_per_col]
            cur_block = block1[:, overlap:]
            lapping = row_block[:, len(row_block[0]) - overlap:]
            lapping1 = block1[:, :overlap]
            for k in range(0, overlap):
                lapping[:, k] *= 1 - (k+1)/(overlap+1)
                lapping1[:, k] *= (k+1)/(overlap+1)
            lap = lapping + lapping1
            row_block = np.concatenate([cur_row_block, lap, cur_block], axis=1)
        return row_block

    img = get_row_block(0)
    for i in range(1, num_block_per_col):
        cur_block = img[:len(img)-overlap]
        cur_row = get_row_block(i)
        lapping = img[len(img)-overlap:]
        lapping1 = cur_row[:overlap]
        cur_row = cur_row[overlap:]
        for k in range(0, overlap):
            lapping[k,:] *= 1 - (k+1)/(overlap+1)
            lapping1[k,:] *= (k+1)/(overlap+1)
        lap = lapping + lapping1
        img = np.concatenate([cur_block, lap, cur_row], axis=0)
    return img

def gen_train_set(clear_imgs, blur_imgs, shape, block_size, num_block, overlap):
    noise_images = np.expand_dims(np.zeros(shape), 0)
    clear_images = np.expand_dims(np.zeros(shape), 0)

    for i in range(len(clear_imgs)):
        blocks = divide_img(clear_imgs[i], block_size, num_block, overlap)
        clear_images = np.concatenate([clear_images, blocks])
        blur_blocks = divide_img(blur_imgs[i], block_size, num_block, overlap)
        noise_images = np.concatenate([noise_images, blur_blocks])
    return clear_images[1:]/255, noise_images[1:]/255

'''
DECODE AND RECONSTRUCT IMAGES
'''
def reconstruct_image(noise_blocks, model, batch_size, block_per_image, width, height, block_size, overlap):
    recons_images = []
    decoded_images = model(noise_blocks[:batch_size])
    for i in range(batch_size, len(noise_blocks), batch_size):
        decoded_images = np.concatenate([decoded_images, 
                                         model(noise_blocks[i:i+batch_size])], 
                                        axis=0)
    blocks = decoded_images[: block_per_image]
    image = merge_img(blocks, width, height, block_size, overlap)
    recons_images = tf.convert_to_tensor([image], np.float32)
    for i in range(block_per_image, len(decoded_images), block_per_image):
        blocks = decoded_images[i: i+block_per_image]
        image = merge_img(blocks, width, height, block_size, overlap)
        recons_images = tf.concat([recons_images, tf.convert_to_tensor([image], np.float32)], axis=0)
    recons_images = tf.cast((recons_images*255), dtype=tf.uint8)
    return recons_images.numpy()

'''
SAVE AND LOAD MODEL
'''
def save_models(model, file_path):
    model.encoder.save(cwd + file_path + 'encoder')
    model.decoder.save(cwd + file_path + 'decoder')
    model.transform.save(cwd + file_path + 'transform')

def load_models(file_path, latent_dim, shape):
    encoder = keras.models.load_model(cwd + file_path + 'encoder')
    decoder = keras.models.load_model(cwd + file_path + 'decoder')
    transform = keras.models.load_model(cwd + file_path + 'transform')
    model = VAE(latent_dim, shape)
    model.encoder = encoder
    model.decoder = decoder
    model.transform = transform
    return model

'''
PLOT IMAGES
'''
def plot_latent(encoder, noise, clean):
    batch = 10000
    x_n,_ = encoder(noise[:batch])
    x_c,_ = encoder(clean[:batch])
    for i in range(batch, len(noise), batch):
        new_x,_ = encoder(noise[i: i+batch])
        x_n = np.concatenate([x_n, new_x], axis=0)

        new_x,_ = encoder(clean[i: i+batch])
        x_c = np.concatenate([x_c, new_x], axis=0)
    pca =  manifold.TSNE(n_components=2)
    x_n = pca.fit_transform(x_n)
    x_c = pca.fit_transform(x_c)
    colors = ['blue', 'red', 'green', 'black', 'yellow', 'purple']
    plt.scatter(x_n[:, 0], x_n[:, 1], s=5, c=colors[0])
    plt.scatter(x_c[:, 0], x_c[:, 1], s=5, c=colors[1])
    plt.show()

def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=20, interpolation='lanczos'):
    images_np = np.swapaxes(np.swapaxes(images_np, 1, 3), 2,3)
    n_channels = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    plt.show()

def display_yuv(img, y, u, v):
    def make_lut_u():
        return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

    def make_lut_v():
        return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
    lut_u, lut_v = make_lut_u(), make_lut_v()

    # Convert back to BGR so we can apply the LUT and stack the images
    if len(y.shape)>2 and len(y[0,0]) > 1:
        y = y[:,:,0]
        u = u[:,:,1]
        v = v[:,:,2]
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)
    images = np.array([img, y, u_mapped, v_mapped])
    plot_image_grid(images, 4)

def display_only_yuv(y, u, v):
    def make_lut_u():
        return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

    def make_lut_v():
        return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
    lut_u, lut_v = make_lut_u(), make_lut_v()

    # Convert back to BGR so we can apply the LUT and stack the images
    if len(y.shape)>2 and len(y[0,0]) > 1:
        y = y[:,:,0]
        u = u[:,:,1]
        v = v[:,:,2]
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)
    images = np.array([y, u_mapped, v_mapped])
    plot_image_grid(images, 4)

def display_patch_matching(noise_ys, noise_us, noise_vs, clear_ys, clear_us, clear_vs, k):
    yy = np.array(noise_ys[k]*255, dtype=np.uint8)
    uu = np.array(noise_us[k]*255, dtype=np.uint8)
    vv = np.array(noise_vs[k]*255, dtype=np.uint8)
    yc = np.array(clear_ys[k]*255, dtype=np.uint8)
    uc = np.array(clear_us[k]*255, dtype=np.uint8)
    vc = np.array(clear_vs[k]*255, dtype=np.uint8)
    y_ = np.array((noise_ys[k] - clear_ys[k])*255, dtype=np.uint8)
    u_ = np.array((noise_us[k] - clear_us[k])*255, dtype=np.uint8)
    v_ = np.array((noise_vs[k] - clear_vs[k])*255, dtype=np.uint8)
    display_only_yuv(yy, uu, vv)
    display_only_yuv(yc, uc, vc)
    display_only_yuv(y_, u_, v_)
