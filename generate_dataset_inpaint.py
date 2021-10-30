import sys, os, glob, time, pdb, cv2
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import shutil
import random
import string

from resize_image import resize_image
import config as cfg
from PIL import ImageFont, ImageDraw, Image  

import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

from faker import Faker
fake = Faker()

def q(text = ''):
    print(f'>{text}<')
    sys.exit()

data_dir = cfg.data_dir
train_dir = cfg.train_dir
val_dir = cfg.val_dir

imgs_dir = cfg.imgs_dir
noisy_dir = cfg.noisy_dir
debug_dir = cfg.debug_dir
patch_dir = cfg.patch_dir

train_data_dir = os.path.join(data_dir, train_dir)
val_data_dir = os.path.join(data_dir, val_dir)

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)

if not os.path.exists(val_data_dir):
    os.mkdir(val_data_dir)

img_train_dir = os.path.join(data_dir, train_dir, imgs_dir)
noisy_train_dir = os.path.join(data_dir, train_dir, noisy_dir)
debug_train_dir = os.path.join(data_dir, train_dir, debug_dir)

img_val_dir = os.path.join(data_dir, val_dir, imgs_dir)
noisy_val_dir = os.path.join(data_dir, val_dir, noisy_dir)
debug_val_dir = os.path.join(data_dir, val_dir, debug_dir)

dir_list = [img_train_dir, noisy_train_dir, debug_train_dir, img_val_dir, noisy_val_dir, debug_val_dir]
for dir_path in dir_list:
    print(f'dir_path = {dir_path}')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def get_word_list():
    f = open(cfg.txt_file_dir, encoding='utf-8', mode="r")
    text = f.read()
    f.close()
    lines_list = str.split(text, '\n')
    while '' in lines_list:
        lines_list.remove('')

    lines_word_list = [str.split(line) for line in lines_list]
    words_list = [words for sublist in lines_word_list for words in sublist]

    return words_list


def __scale_width(img, long_side):
    size = img.shape[:2]
    oh,ow = size
    ratio = oh / ow
    new_width = long_side
    new_height = int(ratio * new_width)

    return cv2.resize(img, (new_width, new_height),interpolation = cv2.INTER_CUBIC)


def get_patches():
    patches = []
    for filename in os.listdir(patch_dir):
        try:
            img_path = os.path.join(patch_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = __scale_width(img, 1792) # 1792 2048  
            patches.append(img)
        except Exception as e:
            raise e
    return patches

words_list = get_word_list()
patches_list = get_patches()

print('\nnumber of words in the txt file: ', len(words_list))

# scale factor
scale_h, scale_w = 1, 1

img_count = 1
word_count = 0
num_imgs = int(cfg.num_synthetic_imgs) # max number of synthetic images to be generated
train_num = int(num_imgs*cfg.train_percentage) # training percent
print('\nnum_imgs : ', num_imgs)
print('train_num: ', train_num)

def get_text():
    global word_count, words_list
    # text to be printed on the blank image
    num_words = np.random.randint(1, 6)
    
    # renew the word list in case we run out of words 
    if (word_count + num_words) >= len(words_list):
        print('===\nrecycling the words_list')
        words_list = get_word_list() 
        word_count = 0

    print_text = ''
    for _ in range(num_words):
        index = np.random.randint(0, len(words_list))
        print_text += str.split(words_list[index])[0] + ' '
        #print_text += str.split(words_list[word_count])[0] + ' '
        word_count += 1
    print_text = print_text.strip() # to get rif of the last space
    return print_text

def get_phone():
    "Generate phone like string"

    letters = string.digits 
    sep = np.random.choice([True, False], p =[0.5, 0.5])
    c = 10
    if sep:
        c = 3
        d = 3
        z = 4
    
    n = (''.join(random.choice(letters) for i in range(c)))
    if sep:
        n += '-'
        n += ''.join(random.choice(letters) for i in range(d))
        n += '-'
        n += ''.join(random.choice(letters) for i in range(z))

    return n


def drawTrueTypeTextOnImage(cv2Image, text, xy, size):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    """
    # Pass the image to PIL  
    pil_im = Image.fromarray(cv2Image)  
    draw = ImageDraw.Draw(pil_im)  

    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "oldfax.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Old_Rubber_Stamp.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "ColourMePurple.ttf", "Pelkistettyatodellisuutta.ttf" ,"SpotlightTypewriterNC.ttf"]) 
    fontPath = os.path.join("./assets/fonts/truetype", fontFace)

    font = ImageFont.truetype(fontPath, size)
    size_width, size_height = draw.textsize(text, font)

    # text has to be within the bounds otherwise return same image
    x = xy[0]
    y = xy[1]
    
    img_h = cv2Image.shape[0]
    img_w = cv2Image.shape[1]

    adj_y = y + size_height
    adj_w = x + size_width
    
    # print(f'size : {img_h},  {adj_y},  {size_width}, {size_height} : {xy}')
    if adj_y > img_h or adj_w > img_w:
        return False, cv2Image, (0, 0)

    draw.text(xy, text, font=font)  
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return True, cv2Image, (size_width, size_height)

def print_lines_single(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # get a line of text
    txt = get_text()
    txt = fake.name()
    # txt = get_phone()

    if np.random.choice([0, 1], p = [0.5, 0.5]) :
        txt = get_text()
        txt = fake.name()
    else:
        txt =  get_phone()  #fake.address()

    # letters = string.digits 
    # c = np.random.randint(4, 9)
    # txt = (''.join(random.choice(letters) for i in range(c)))
    # txt = get_phone() 

    if np.random.choice([0, 1], p = [0.5, 0.5]):
        txt = txt.upper()
            
    txt =  getUpperOrLowerText(txt)
    trueTypeFontSize = np.random.randint(45, 60)

    # img = drawTrueTypeTextOnImage(img, txt, (np.random.randint(0, img.shape[1] / 4), np.random.randint(img.shape[0]/3, img.shape[0])), trueTypeFontSize)
    img = drawTrueTypeTextOnImage(img, txt, (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])), trueTypeFontSize)

    return img

def print_lines_aligned(img, boxes):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    def make_txt():
        # get a line of text
        txt = get_text()

        if np.random.choice([0, 1], p = [0.8, 0.2]) :
            txt = get_text()
            # txt = fake.name()
        else:
            # txt =  get_phone()  #fake.address()
            letters = string.digits 
            c = np.random.randint(1, 9)
            txt = (''.join(random.choice(letters) for i in range(c)))

        # letters = '171717171717171717'
        # c = np.random.randint(1, 9)
        # txt = (''.join(random.choice(letters) for i in range(c)))
        
        if np.random.choice([0, 1], p = [0.5, 0.5]):
            txt = txt.upper()

        return getUpperOrLowerText(txt)
   
    trueTypeFontSize = np.random.randint(28, 42)
    xy = (np.random.randint(0, img.shape[1] / 10), np.random.randint(0, img.shape[0] / 8))

    w = img.shape[1]
    h = img.shape[0]
    start_y = xy[1]

    while True:
        m_h = np.random.randint(60, 120)
        start_x = xy[0]
        while True:
            txt = make_txt()    
            pos = (start_x, start_y)
            valid, img, wh = drawTrueTypeTextOnImage(img, txt, pos, trueTypeFontSize)
            txt_w =  wh[0] + np.random.randint(60, 120)
            # print(f' {start_x}, {start_y} : {valid}  : {wh}')
            start_x = start_x + txt_w
            if wh[1] > m_h:
                m_h = wh[1]
            # break
            if start_x > w:
                break
        start_y = start_y +  np.random.randint(m_h//2, m_h*1.5)
        if start_y > h:
            break
        
    box = [xy[0], xy[1], wh[0], wh[1]]
    boxes.append(box)

    return True, img


def print_lines(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # get a line of text
    if np.random.choice([0, 1], p = [0.5, 0.5]) :
        txt = get_text()
    else:
        letters = string.digits 
        c = np.random.randint(1, 9)
        txt = (''.join(random.choice(letters) for i in range(c)))

    if np.random.choice([0, 1], p = [0.5, 0.5]):
        txt = txt.upper()

    # # txt = fake.name()
    # letters = string.ascii_lowercase 
    # c = np.random.randint(1, 4)
    # txt = (''.join(random.choice(letters) for i in range(c)))

    # txt = get_phone()
    txt = fake.name()
    # txt = fake.address()
    txt = getUpperOrLowerText(txt)

    trueTypeFontSize = np.random.randint(32, 62)
    valid, img, _ = drawTrueTypeTextOnImage(img, txt, (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])), trueTypeFontSize)
    
    return valid, img


def get_debug_image(img, noisy_img):
    debug_img = np.ones((2*h, w), dtype = np.uint8)*255 # to visualize the generated images (clean and noisy)
    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = noisy_img
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)
    return debug_img


def write_images(img, noisy_img, debug_img, index):
    img_type = ''
    print(f'Writing {index}, {train_num}')

    if index <= train_num:            
        cv2.imwrite(os.path.join(data_dir, train_dir, imgs_dir, '{}.png'.format(str(index).zfill(8), img_type)), noisy_img ) 
        cv2.imwrite(os.path.join(data_dir, train_dir, noisy_dir, '{}.png'.format(str(index).zfill(8), img_type)), img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, debug_dir, '{}.png'.format(str(index).zfill(8),img_type)), debug_img) 
    else:
        cv2.imwrite(os.path.join(data_dir, val_dir, imgs_dir, '{}.png'.format(str(index).zfill(8),img_type)), noisy_img ) 
        cv2.imwrite(os.path.join(data_dir, val_dir, noisy_dir, '{}.png'.format(str(index).zfill(8),img_type)), img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, debug_dir, '{}.png'.format(str(index).zfill(8),img_type)), debug_img) 

print('\nsynthesizing image data...')
idx = 0

def __process(index):
    print(f'index : {index}')
    try:
        patch_idx = np.random.randint(0, len(patches_list))
        patch = patches_list[patch_idx]
        h = patch.shape[0]
        w = patch.shape[1]

        # make a blank image
        img = np.ones((h, w), dtype = np.uint8) * 255
        boxes = []
        # try to acquire an image 
        while True:
            valid, img = print_lines_aligned(img, boxes)
            if valid:
                break
        
        # turn black/white into a grayscale mask 
        mask = patch.copy()
        if True:
            mask[mask >= 230] = [235]
            mask[mask < 230] = [255]

        # # write images
        # print(f'idx/patch_idx = {idx} , {patch_idx}')

        # data_dir = '/tmp/form-segmentation'
        # debug_dir = 'debug'
        # img_dir = 'image'
        # mask_dir = 'mask'
        
        kernel = np.ones((4, 4), np.uint8)
        img_erode = cv2.erode(img, kernel, iterations=1)

        patch = cv2.bitwise_and(patch, img, mask = None)
        patch = cv2.threshold(patch, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        __img = cv2.threshold(img_erode, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # __img = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        mask = cv2.bitwise_and(mask, __img, mask = None)
        # mask = cv2.bitwise_and(mask, img, mask = None)
        
        write_images(patch, mask, img, index)
        # write_images(mask, img, img, index)
        # write_images(mask, patch, img, index)
    except Exception as e:
        raise e
        print(e)

# fireup new threads for processing
with ThreadPoolExecutor(max_workers=mp.cpu_count()* 2) as executor:
    for i in range(0, num_imgs):
        executor.submit(__process, i)

print('All tasks has been finished')
