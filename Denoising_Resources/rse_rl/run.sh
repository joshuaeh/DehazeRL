#!/usr/bin/env bash

python3 experiment_celeba.py --file_batch 5 --epoch 50 --latent_dim 72 --target_psnr 28 --train_files_path "../data/celeba_train/" --test_files_path "../data/celeba_test/"