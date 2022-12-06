import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--block_size', type=int, default=16)
parser.add_argument('--num_block_per_row', type=int, default=21)
parser.add_argument('--overlap', type=int, default=4)
parser.add_argument('--file_batch', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=96)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--train_files_path', type=str, default='')
parser.add_argument('--test_files_path', type=str, default='')
parser.add_argument('--test_validation_files_path', type=str, default='')
parser.add_argument('--target_psnr', type=float, default=34.0)
parser.add_argument('--plot', type=bool, default=False)

from models import *
from evaluation import *
from RL import *
    

def main(args):
    # Enable GPU
    if args.gpu:
        #%tensorflow_version 2.x
        import tensorflow as tf
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
          raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))


    BLOCK_SIZE = args.block_size
    NUM_BLOCK = args.num_block_per_row
    BLOCK_PER_IMAGE = NUM_BLOCK * NUM_BLOCK
    OVERLAP = args.overlap
    SHAPE = (BLOCK_SIZE, BLOCK_SIZE, 1)
    LATENT_DIM = args.latent_dim
    EPOCH = args.epoch
    FILE_BATCH = args.file_batch

    '''
    Load Images
    '''
    train_files = []
    valid_files = []
    for root_path, sub, files in os.walk(args.train_files_path):
        contents = files
        contents.sort()
        for f in contents:
            file_path = os.path.join(root_path,f)
            if os.path.isfile(file_path) and "GT" in f:
                train_files.append(file_path)
            if os.path.isfile(file_path) and "NOISY" in f:
                valid_files.append(file_path)
    if len(train_files) != len(valid_files):
        raise ValueError('Train and Validation file must have same length', len(train_files), len(valid_files))
    
    '''
    Build Networks
    '''
    model_y = VAE(LATENT_DIM, SHAPE)
    model_u = VAE(LATENT_DIM, SHAPE)
    model_v = VAE(LATENT_DIM, SHAPE)

    '''
    Train Network
    '''
    for fb in range(0, len(train_files), FILE_BATCH):
        train = train_files[fb:fb+FILE_BATCH]
        valid = valid_files[fb:fb+FILE_BATCH]
        clear_images = crop_square(read_images(valid))
        noise_images = crop_square(read_images(train))
        WIDTH = len(clear_images[0][0])
        HEIGHT = len(clear_images[0])

        clear_ys, clear_us, clear_vs = cvt_bgr_yuv(clear_images)
        noise_ys, noise_us, noise_vs = cvt_bgr_yuv(noise_images)
        clear_ys, noise_ys = gen_train_set(clear_ys, noise_ys, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_us, noise_us = gen_train_set(clear_us, noise_us, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_vs, noise_vs = gen_train_set(clear_vs, noise_vs, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)

        train_model(model_y, clear_ys, noise_ys, EPOCH) 
        train_model(model_u, clear_us, noise_us, EPOCH)
        train_model(model_v, clear_vs, noise_vs, EPOCH)

    save_models(model_y, 'pretrained models/sidd_y/')
    save_models(model_u, 'pretrained models/sidd_u/')
    save_models(model_v, 'pretrained models/sidd_v/')

    '''
    Evaluation
    '''
    avg_psnr, avg_ssim, avg_uqi = 0, 0, 0
    BATCH = 4
    NUM_BLOCK = 15

    for i in range(BATCH):
        clear_images = sidd_test_data(args.test_validation_files_path, 'ValidationGtBlocksSrgb', i)
        test_images = np.array(clear_images)
        noise_images = sidd_test_data(args.test_files_path, 'ValidationNoisyBlocksSrgb', i)
        WIDTH = len(clear_images[0][0])
        HEIGHT = len(clear_images[0])

        test_images = np.array(clear_images)
        clear_ys, clear_us, clear_vs = cvt_bgr_yuv(clear_images)
        noise_ys, noise_us, noise_vs = cvt_bgr_yuv(noise_images)
        
        clear_ys, noise_ys = gen_train_set(clear_ys, noise_ys, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_us, noise_us = gen_train_set(clear_us, noise_us, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        clear_vs, noise_vs = gen_train_set(clear_vs, noise_vs, SHAPE, BLOCK_SIZE, NUM_BLOCK, OVERLAP)
        
        recons_ys = reconstruct_image(noise_ys, model_y, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_us = reconstruct_image(noise_us, model_u, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)
        recons_vs = reconstruct_image(noise_vs, model_v, BATCH_SIZE, BLOCK_PER_IMAGE, WIDTH, HEIGHT, BLOCK_SIZE, OVERLAP)

        recons_images = cvt_yuv_bgr(recons_ys, recons_us, recons_vs)

        if args.plot:
            display_patch_matching(noise_ys, noise_us, noise_vs, clear_ys, clear_us, clear_vs, 0)
            display_yuv(noise_images[0], noise_ys[0], noise_us[0], noise_vs[0])
            display_yuv(recons_images[0], recons_ys[0], recons_us[0], recons_vs[0])
            plot_image_grid(recons_images[:4], 4)

        avg_psnr += quality_evaluation(recons_images, test_images, metric='PSNR')
        avg_ssim += quality_evaluation(recons_images, test_images, metric='SSIM')
        avg_uqi += quality_evaluation(recons_images, test_images, metric='UQI')
    print('***********************')
    print('Overall Results')
    print('PSNR: ', avg_psnr/BATCH)
    print('SSIM: ', avg_ssim/BATCH)
    print('UQI: ', avg_uqi/BATCH)
    print('***********************')

    '''
    Self-Enhancing
    '''
    TARGET_PSNR = args.target_psnr

    for fb in range(0, len(train_files), FILE_BATCH):
        train = train_files[fb:fb+FILE_BATCH]
        valid = valid_files[fb:fb+FILE_BATCH]
        clear_images = crop_square(read_images(valid))
        noise_images = crop_square(read_images(train))

        env = DummyVecEnv([lambda: Img_Enhancing_Env(clear_images, noise_images, [model_y, model_u, model_v], TARGET_PSNR)])
        rl_model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)
        rl_model.learn(total_timesteps=100)

if __name__ == '__main__':
    main(parser.parse_args())
