import os

# path to the data directories
data_dir = 'data'
train_dir = 'train'
val_dir = 'test'
imgs_dir = 'mask'
noisy_dir = 'image'
debug_dir = 'debug'
patch_dir = './assets/patches'
# patch_dir = './assets/field-patches/export'

# text file to get text from
txt_file_dir = 'text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 10000
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)