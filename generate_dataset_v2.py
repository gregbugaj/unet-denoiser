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

def get_patches():
    patches = []
    for filename in os.listdir(patch_dir):
        try:
            img_path = os.path.join(patch_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            patches.append(img)
        except Exception as e:
            print(e)
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
    # use a truetype font  
    # /usr/share/fonts/truetype/freefont/FreeSerif.ttf"
    # FreeMono.ttf
    # FreeSerif.ttf
    # "FreeSerif.ttf","FreeSerifBold.ttf",
    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf"]) 
    # fontPath = os.path.join("/usr/share/fonts/truetype/freefont", fontFace)

    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "oldfax.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Old_Rubber_Stamp.ttf"]) 
    # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf",  "FreeSans.ttf", "ColourMePurple.ttf", "Pelkistettyatodellisuutta.ttf" ,"SpotlightTypewriterNC.ttf"]) 
    fontPath = os.path.join("./assets/fonts/truetype", fontFace)

    font = ImageFont.truetype(fontPath, size)
    draw.text(xy, text, font=font)  
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)
    return cv2Image

def print_lines_single(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # get a line of text
    txt = get_text()
    txt = fake.name()
    txt =  getUpperOrLowerText(txt)
    trueTypeFontSize = np.random.randint(40, 52)

    img = drawTrueTypeTextOnImage(img, txt, (np.random.randint(-5, img.shape[1] ), np.random.randint(20, img.shape[0])), trueTypeFontSize)

    return img

def print_lines_DIAGNOSIS_CODE(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    def get_text_x():
        # get a line of text
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            txt = get_text()
        else:
            txt = fake.name().split(' ')[0]
        return txt

    # get a line of text
    txt = get_text()
    # txt = fake.name().split(' ')[0]
    # txt =  getUpperOrLowerText(txt)
    trueTypeFontSize = np.random.randint(40, 52)

    rows = 6
    x1 = img.shape[1] / 6
    y1 = img.shape[0] / rows

    px = np.random.randint(0, x1)
    p2 = np.random.randint(0, y1)

    for i in range(0, rows):
        p2 = p2 + y1
        p1 = px

        txt = get_text_x()
        txt =  getUpperOrLowerText(txt)
        img = drawTrueTypeTextOnImage(img, txt, (p1 , p2), trueTypeFontSize)

        txt = get_text_x()
        txt =  getUpperOrLowerText(txt)
        p1 = np.random.randint(p1 + x1, p1 + x1 * 2)
        img = drawTrueTypeTextOnImage(img, txt, (p1 , p2), trueTypeFontSize)    
        
        txt = get_text_x()
        txt =  getUpperOrLowerText(txt)
        p1 = np.random.randint(p1 + x1, p1 + x1 * 3)
        img = drawTrueTypeTextOnImage(img, txt, (p1 , p2), trueTypeFontSize)

        txt = get_text_x()
        txt =  getUpperOrLowerText(txt)
        p1 = np.random.randint(p1 + x1, p1 + x1 * 4)
        img = drawTrueTypeTextOnImage(img, txt, (p1 , p2), trueTypeFontSize)

    return img


def print_lines(img):
    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # phones are somewhat fixed at specific locations 
    # box 33
    trueTypeFontSize = np.random.randint(38, 52)
    phone = get_phone()            
    img = drawTrueTypeTextOnImage(img, phone, (np.random.randint(500, 550), np.random.randint(-10, 55)), trueTypeFontSize)
        
    # get a line of text
    if np.random.choice([0, 1], p = [0.5, 0.5]) :
        txt = get_text()
    else:
        txt = fake.name()

    name = fake.name()
    address = fake.address()
    txt = "{}\n{}".format(name, address)
    if np.random.choice([0, 1], p = [0.5, 0.5]):
        txt = txt.upper()
            
    txt =  getUpperOrLowerText(txt)
    trueTypeFontSize = np.random.randint(40, 52)

    img = drawTrueTypeTextOnImage(img, txt, (np.random.randint(-10, img.shape[1] / 3), np.random.randint(30, img.shape[0] / 3)), trueTypeFontSize)
    return img


def get_debug_image(img, noisy_img):
    debug_img = np.ones((2*h, w), dtype = np.uint8)*255 # to visualize the generated images (clean and noisy)
    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = noisy_img
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)
    return debug_img


def write_images(img, noisy_img, debug_img):
    global img_count
    # img       = 255 - cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)
    # noisy_img = 255 - cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    # debug_img = 255 - cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)    
    
    # img       = 255 - cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)

    scale_h = np.random.uniform(.9, 1.1)
    scale_w = np.random.uniform(.9, 1.1)

    if  np.random.choice([True, False], p = [0.65, 0.35]):
       w = img.shape[1] + np.random.randint(0, 30)
       h = img.shape[0] + np.random.randint(0, 30)
       img =  resize_image(img, (h, w), color=(255, 255, 255))
       noisy_img = resize_image(noisy_img, (h, w), color=(255, 255, 255))

    img       = 255 - cv2.resize(img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    noisy_img = cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    debug_img = cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_type = 'overflow'

    if img_count <= train_num:            
        cv2.imwrite(os.path.join(data_dir, train_dir, imgs_dir, '{}_{}.png'.format(str(img_count).zfill(8), img_type)), img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, noisy_dir, '{}_{}.png'.format(str(img_count).zfill(8), img_type)), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, debug_dir, '{}_{}.png'.format(str(img_count).zfill(8),img_type)), debug_img) 
    else:
        cv2.imwrite(os.path.join(data_dir, val_dir, imgs_dir, '{}_{}.png'.format(str(img_count).zfill(8),img_type)), img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, noisy_dir, '{}_{}.png'.format(str(img_count).zfill(8),img_type)), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, debug_dir, '{}_{}.png'.format(str(img_count).zfill(8),img_type)), debug_img) 

    img_count += 1


print('\nsynthesizing image data...')
for i in tqdm(range(num_imgs)):
    try:
        patch = patches_list[np.random.randint(0, len(patches_list))]
        h = patch.shape[0]
        w = patch.shape[1]

        # make a blank image
        img = np.ones((h, w), dtype = np.uint8) * 255

        # put text
        # img = print_lines(img)
        img = print_lines_DIAGNOSIS_CODE(img)
        noisy_img = cv2.bitwise_and(patch, img, mask = None)

        # # make debug image
        debug_img = get_debug_image(img, noisy_img)

        # # write images
        write_images(img, noisy_img, debug_img)
    except Exception as e:
        print(e)

    '''
    cv2.imshow('textonimage', original)
    cv2.imshow('noisy_img', noisy_img)
    cv2.waitKey()
    '''