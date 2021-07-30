import os

# path to the data directories
data_dir = 'data-patches-2'
# data_dir = 'data-val-box33-set1'
# data_dir = 'data-val-patches'
data_dir = 'data-val-patches-2' # proc codes
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

# Procedure codes 128,352
patch_dir = './assets/patches-2'
txt_file_dir = 'text.txt'


# Insured codes 128,1056
data_dir = 'data' # proc codes
train_dir = 'train'
patch_dir = '/home/greg/TRAINING-ON-DD-GPU/gpu/training/INSURED_ID/images/redacted/scaled'
txt_file_dir = 'text.txt'

# Diagnosis codes 128,352
# data_dir = 'data'
# # data_dir = 'data-val-DIAGNOSIS_CODE_SELECTED-01'
# train_dir = 'train'
# patch_dir = '/home/greg/TRAINING-ON-DD-GPU/gpu/training/DIAGNOSIS_CODE_SELECTED/images/padded'
# txt_file_dir = 'text.txt'


# # box 31
# patch_dir = './assets/patches-3/box31CleanedImages/box31'
# txt_file_dir = 'text.txt'


# # box 33
patch_dir = './assets/patches-3/box33CleanedImages/box33'
txt_file_dir = 'text.txt'

patch_dir = './assets/backgrounds/diagnosis_code'
txt_file_dir = 'text.txt'

patch_dir = './assets/backgrounds/service_lines'
txt_file_dir = 'text.txt'

# patch_dir = './assets/box_2_a'
# txt_file_dir = 'text.txt'

# HCFA07Phone
# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA07Phone'
# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA05PatientAddressOne'

# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/CITY'
# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/STATE'
# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/ZIP'
txt_file_dir = 'text.txt'


# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA24NoText'
# txt_file_dir = 'text.txt'

# # Diagnosis
# # patch_dir = './assets/patches'
# # txt_file_dir = 'text.txt'

# patch_dir = './assets/patches-3/box33CleanedImages/box33'
# txt_file_dir = 'text.txt'

patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA24NoText/mod-v2'
# # patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA24NoText/patches'
# # txt_file_dir = 'text.txt'# 
# # 
# patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA02'

patch_dir = '/home/greg/dev/unet-denoiser/assets/backgrounds/diagnosis_code'
patch_dir = '/home/greg/dev/unet-denoiser/assets/backgrounds/diagnosis_code_alpha'

patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA04'
patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/HCFA24NoText'
patch_dir = '/home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/formsnippets/STATE'
txt_file_dir = 'text.txt'


# maximun number of synthetic words to generate
num_synthetic_imgs = 200
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)

