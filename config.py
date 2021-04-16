import os

# path to the data directories
data_dir = 'data-val-box31'
train_dir = 'train'

val_dir = 'test'
imgs_dir = 'mask'

noisy_dir = 'image'
debug_dir = 'debug'

patch_dir = './assets/patches'
txt_file_dir = 'text.txt'

# patch_dir = './assets/field-patches/export'
# txt_file_dir = 'digits.txt'

# text file to get text from

patch_dir = './assets/patches'
txt_file_dir = 'text.txt'

patch_dir = './assets/patches-3/box31CleanedImages/box31'
txt_file_dir = 'text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 100
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)