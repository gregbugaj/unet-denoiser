import sys, os, glob, time, pdb, cv2
import numpy as np
from numpy.lib.function_base import angle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import shutil
import random
import string

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

def resize_image(image, desired_size, color=(255, 255, 255)):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------
    
    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    '''

    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0])/size[0]
        ratio_h = float(desired_size[1])/size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x*ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image


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
            
            # add extra padding so we have some room to expand text beyond the path
            img = resize_image(img, (320, 1200))
            patches.append(img)
        except Exception as e:
            print(e)
    return patches

words_list = get_word_list()
patches_list = get_patches()

print('\nnumber of words in the txt file: ', len(words_list))

# list of all the font styles
font_list = [
            # cv2.FONT_HERSHEY_COMPLEX, 
            #  cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #  cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_PLAIN,
             cv2.FONT_HERSHEY_SIMPLEX,
            #  cv2.FONT_HERSHEY_TRIPLEX,
             #cv2.FONT_ITALIC
             ] # cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cursive

# size of the synthetic images to be generated
syn_h, syn_w = 128, 352 # PROD 
syn_h, syn_w = 350, 700 # PROD box31
syn_h, syn_w = 300, 1000 # PROD box33
syn_h, syn_w = 256, 1024 # PROD box33

# syn_h, syn_w = 120, 600 # PROD-SEGMENTS
#$syn_h, syn_w = 220, 1500 # PROD-SEGMENTS

# scale factor
scale_h, scale_w = 1, 1

# initial size of the image, scaled up by the factor of scale_h and scale_w
h, w = syn_h*scale_h, syn_w*scale_w 

img_count = 1
word_count = 0
num_imgs = int(cfg.num_synthetic_imgs) # max number of synthetic images to be generated
train_num = int(num_imgs*cfg.train_percentage) # training percent
print('\nnum_imgs : ', num_imgs)
print('train_num: ', train_num)

word_start_x = 10 # min space left on the left side of the printed text
word_start_y = 1
word_end_y = 1   # min space left on the bottom side of the printed text

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

def get_text():
    global word_count, words_list
    # text to be printed on the blank image
    num_words = np.random.randint(2, 6)
    
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

def get_text_height(img, fontColor):
    black_coords = np.where(img == fontColor)
    # finding the extremes of the printed text
    ymin = np.min(black_coords[0])
    ymax = np.max(black_coords[0])
    # xmin = np.min(black_coords[1])
    # xmax = np.max(black_coords[1])
    ''' # for vizualising
    cv2.line(img, (0,ymin),(1000,ymin), 0,2)
    cv2.line(img, (0,ymax),(1000,ymax), 0,2)
    cv2.imshow('ymax', img)
    '''
    return ymax - ymin


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

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
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Times New Roman 400.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Times New Roman 400.ttf", "Fancy Signature Extras.ttf"]) 
    fontFace = np.random.choice([ "FreeMono.ttf",  "FreeSans.ttf", "ColourMePurple.ttf", "Pelkistettyatodellisuutta.ttf" ,"SpotlightTypewriterNC.ttf"]) 
    fontPath = os.path.join("./assets/fonts/truetype", fontFace)

    font = ImageFont.truetype(fontPath, size)
    draw.text(xy, text, font=font)  
    # Make Numpy/OpenCV-compatible version
    cv2Image = np.array(pil_im)

    # Degrade font quality 
    # res1 = rescale_frame(cv2Image, np.random.randint(30, 80))
    width = cv2Image.shape[1] 
    height = cv2Image.shape[0] 
    dim = (width, height)

    # noise = cv2.resize(res1, dim, interpolation = cv2.INTER_AREA)
    noise = cv2.threshold(cv2Image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return noise

def print_lines(img, font, bottomLeftCornerOfText, fontColor, fontScale, lineType, thickness):
    
    line_num = 0
    y_line_list = []

    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p = [0.5, 0.5]) :
            return txt.upper()
        return txt.lower()    

    # print('img.shape: ', img.shape)
    print('initial bottomLeftCornerOfText: ', bottomLeftCornerOfText)
    fontColor = 0
    while bottomLeftCornerOfText[1] <= img.shape[0]:
        # get a line of text
        print_text = get_text()
        txt =  getUpperOrLowerText(print_text)

        # phones are somewhat fixed at specific locations 
        # box 33
        trueTypeFontSize = np.random.randint(38, 52)
        if line_num == 0:
            phone = get_phone()            
            img = drawTrueTypeTextOnImage(img, phone, (np.random.randint(500, 550), np.random.randint(-10, 55)), trueTypeFontSize)
      
        # put it on a blank image and get its height
        if line_num == 0:
            # get the correct text height 
            
            if False:
                big_img = np.ones((500, 500), dtype = np.uint8)*255
                big_img_text = getUpperOrLowerText(print_text)
                cv2.putText(img = big_img, text = big_img_text, org = (0, 300), fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
                text_height = get_text_height(big_img, fontColor)
                if text_height > bottomLeftCornerOfText[1]:
                    bottomLeftCornerOfText = (bottomLeftCornerOfText[0], np.random.randint(word_start_y, int(img.shape[0]*0.5)) + text_height)

                # Sometime we wan to print TrueType only
                if  True or np.random.choice([True, False], p = [0.50, 0.50]):
                    trueTypeFontSize = np.random.randint(40, 44)
                    img = drawTrueTypeTextOnImage(img, txt, bottomLeftCornerOfText, trueTypeFontSize)
                    # continue
                else:
                    cv2.putText(img = img, text = getUpperOrLowerText(print_text), org = bottomLeftCornerOfText, fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
                
                y_line_list.append(bottomLeftCornerOfText[1])
            name = fake.name()
            address = fake.address()
            spot = "{}\n{}".format(name, address)
            if np.random.choice([0, 1], p = [0.5, 0.5]):
                spot = spot.upper()

            img = drawTrueTypeTextOnImage(img, spot, (np.random.randint(80, 200), np.random.randint(60, 140)), trueTypeFontSize)
            text_height = 100
            y_line_list.append(0)
            break
        else:
            # sampling the chances of adding one more line of text
            one_more_line = np.random.choice([0, 1], p = [0.5, 0.5]) # .3 , .7
            if not one_more_line:
                break
            
            # cv2.putText(img = img, text = getUpperOrLowerText(txt), org = bottomLeftCornerOfText, fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
            trueTypeFontSize = np.random.randint(40, 50)
            img = drawTrueTypeTextOnImage(img, txt, bottomLeftCornerOfText, trueTypeFontSize)

            y_line_list.append(bottomLeftCornerOfText[1])
        # calculate the (text_height+line break space) left on the bottom
        bottom_space_left = int(text_height*(1 + np.random.randint(5, 10)/100))
        # bottom_space_left = int(text_height*(1))
        # print('bottom_space_left: ', bottom_space_left)

        # update the bottomLeftCornerOfText
        bottomLeftCornerOfText = (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + bottom_space_left)
        line_num += 1
    '''    
    for l in y_line_list:
        cv2.line(img, (0, l), (1000, l), 0, 1)
    '''
    # (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img, y_line_list, text_height

        
def get_noisy_img(img, y_line_list, text_height):
    # adding noise (horizontal and vertical lines) on the image containing text
    noisy_img = img.copy()

    #Do we want to make a dirty image
    # if False or np.random.choice([True, False], p = [0.50, 0.50]):
    #     return noisy_img

    # Add background patch
    # if True or np.random.choice([True, False], p = [0.80, 0.20]):
    patch = patches_list[np.random.randint(0, len(patches_list))]
    patch = cv2.resize(patch, (w,h))
    noisy_img = cv2.bitwise_and(patch, noisy_img, mask = None)

    if False and np.random.choice([True, False], p = [0.60, 0.40]):
        # adding horizontal line (noise)
        for y_line in y_line_list: 

            # samples the possibility of adding a horizontal line
            add_horizontal_line = np.random.choice([0, 1], p = [0.5, 0.5])
            if not add_horizontal_line:
                continue

            # shift y_line randomly in the y-axis within a defined limit
            limit = int(text_height*0.3)
            if limit == 0: # this happens when the text used for getting the text height is '-', ',', '=' and other little symbols like these 
                limit = 10
            y_line += np.random.randint(-limit, limit)

            h_start_x = 0 #np.random.randint(0,xmin)                           # min x of the horizontal line
            h_end_x   = np.random.randint(int(noisy_img.shape[1]*0.8), noisy_img.shape[1]) # max x of the horizontal line
            h_length = h_end_x - h_start_x + 1
            num_h_lines = np.random.randint(10,30) # partitions to be made in the horizontal line (necessary to make it look like naturally broken lines)
            h_lines = []
            h_start_temp = h_start_x
            next_line = True

            num_line = 0
            while (next_line) and (num_line < num_h_lines):
                if h_start_temp < h_end_x:
                    h_end_temp = np.random.randint(h_start_temp + 1, h_end_x + 1)
                    if h_end_temp < h_end_x:
                        h_lines.append([h_start_temp, h_end_temp]) 
                        h_start_temp = h_end_temp + 1
                        num_line += 1
                    else:
                        h_lines.append([h_start_temp, h_end_x]) 
                        num_line += 1
                        next_line = False
                else:
                    next_line = False

            for h_line in h_lines:
                col = np.random.choice(['black', 'white'], p = [0.80, 0.20]) # probabilities of line segment being a solid one or a broken one
                if col == 'black':
                    x_points = list(range(h_line[0], h_line[1] + 1))
                    x_points_black_prob = np.random.choice([0,1], size = len(x_points), p = [0.2, 0.8])

                    y_pos = y_line - np.random.randint(4)
                    cv2.line(noisy_img, (0, y_pos), (noisy_img.shape[1], y_pos), (0, 0, 0), 2)
                    if False:
                        for idx, x in enumerate(x_points):
                            if x_points_black_prob[idx]:
                                noisy_img[ y_line - np.random.randint(4): y_line + np.random.randint(4), x] = np.random.randint(0,30)  


    # if np.random.choice([True, False], p = [0.60, 0.40]):
    #     y_pos = np.random.randint(80)
    #     cv2.line(noisy_img, (0, y_pos), (noisy_img.shape[1], y_pos), (0, 0, 0), np.random.randint(1, 3))       
    
    # if np.random.choice([True, False], p = [0.60, 0.40]):
    #     y_pos = np.random.randint(noisy_img.shape[0]-80, noisy_img.shape[0])
    #     cv2.line(noisy_img, (0, y_pos), (noisy_img.shape[1], y_pos), (0, 0, 0), np.random.randint(2, 4))    
    
    # adding vertical line (noise)
    vertical_bool = {'left': np.random.choice([0,1], p =[0.3, 0.7]), 'right': np.random.choice([0,1])} # [1 or 0, 1 or 0] whether to make vertical left line on left and right side of the image
    for left_right, bool_ in vertical_bool.items():
        if bool_:
            # print('left_right: ', left_right)
            if left_right == 'left':
                v_start_x = np.random.randint(5, int(noisy_img.shape[1]*0.06))
            else:
                v_start_x = np.random.randint(int(noisy_img.shape[1]*0.95), noisy_img.shape[1] - 5)

            v_start_y = np.random.randint(0, int(noisy_img.shape[0]*0.06))
            v_end_y   = np.random.randint(int(noisy_img.shape[0]*0.95), noisy_img.shape[0])

            y_points = list(range(v_start_y, v_end_y + 1))
            y_points_black_prob = np.random.choice([0,1], size = len(y_points), p = [0.2, 0.8])

            # cv2.line(noisy_img, (v_start_x, v_start_y), (v_start_x, h), (0, 0, 0), np.random.randint(1, 3))
            
            if False:
                for idx, y in enumerate(y_points):
                    if y_points_black_prob[idx]:
                        noisy_img[y, v_start_x - np.random.randint(2): v_start_x + np.random.randint(2)] = np.random.randint(0,30)

    return noisy_img

def degrade_qualities(img, noisy_img):
    '''
    This function takes in a couple of images (color or grayscale), downsizes it to a
    randomly chosen size and then resizes it to the original size,
    degrading the quality of the images in the process.
    '''

    # if np.random.choice([True, False], p = [0.70, 0.30]):
    #     return img, noisy_img    

    h, w = img.shape[0], img.shape[1]
    fx=np.random.randint(50,100)/100
    fy=np.random.randint(50,100)/100
    # print('fx, fy: ', fx, fy)
    img_small = cv2.resize(img, (0,0), fx = fx, fy = fy) 
    img = cv2.resize(img_small,(w,h))
    
    noisy_img_small = cv2.resize(noisy_img, (0,0), fx = fx, fy = fy) 
    noisy_img = cv2.resize(noisy_img_small,(w,h))

    return img, noisy_img

def get_debug_image(img, noisy_img):
    debug_img = np.ones((2*h, w), dtype = np.uint8)*255 # to visualize the generated images (clean and noisy)
    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = noisy_img
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)
    return debug_img

def erode_dilate(img, noisy_img):

    if np.random.choice([True, False], p = [0.50, 0.50]):
        return img, noisy_img    

    # erode the image
    kernel = np.ones((3,3),np.uint8)
    erosion_iteration = np.random.randint(1,3)
    dilate_iteration = np.random.randint(0,2)
    img = cv2.erode(img,kernel,iterations = erosion_iteration)
    noisy_img = cv2.erode(noisy_img,kernel,iterations = erosion_iteration)
    img = cv2.dilate(img,kernel,iterations = dilate_iteration)
    noisy_img = cv2.dilate(noisy_img,kernel,iterations = dilate_iteration)
    return img, noisy_img

def write_images(img, noisy_img, debug_img):
    global img_count
    # img       = 255 - cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)
    # noisy_img = 255 - cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    # debug_img = 255 - cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)    
    
    # img       = 255 - cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)

    scale_h = np.random.uniform(.9, 1.1)
    scale_w = np.random.uniform(.9, 1.1)

    img       =  cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)
    noisy_img = cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    debug_img = cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # # Frame to target for UNET
    # img = resize_image(img, (1024, 1024))
    # noisy_img = resize_image(noisy_img, (1024, 1024))

    if img_count <= train_num:            
        cv2.imwrite(os.path.join(data_dir, train_dir, imgs_dir, '{}.png'.format(str(img_count).zfill(8))), img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, noisy_dir, '{}.png'.format(str(img_count).zfill(8))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, debug_dir, '{}.png'.format(str(img_count).zfill(8))), debug_img) 
    else:
        cv2.imwrite(os.path.join(data_dir, val_dir, imgs_dir, '{}.png'.format(str(img_count).zfill(8))), img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, noisy_dir, '{}.png'.format(str(img_count).zfill(8))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, debug_dir, '{}.png'.format(str(img_count).zfill(8))), debug_img) 

    img_count += 1

print('\nsynthesizing image data...')
for i in tqdm(range(num_imgs)):
    # make a blank image
    img = np.ones((h, w), dtype = np.uint8) * 255
    # set random parameters
    font = font_list[np.random.randint(len(font_list))]
    # bottomLeftCornerOfText = (np.random.randint(word_start_x, int(img.shape[1]/3)), np.random.randint(0, int(img.shape[0]*0.8))) # (x, y)

    bottomLeftCornerOfText = (np.random.randint(word_start_x, int(img.shape[1]/8)), np.random.randint(0, int(img.shape[0]*0.8))) # (x, y)
    bottomLeftCornerOfText = (np.random.randint(word_start_x, int(img.shape[1]/8)), np.random.randint(int(img.shape[0] / 3), int(img.shape[0] / 3) + int(img.shape[0]*0.2))) # (x, y)


    # fontColor              = np.random.randint(0, 30)
    fontColor              = 0 # np.random.randint(2)
    fontScale              = np.random.randint(2300, 2400)/ 2400
    fontScale              = np.random.uniform(1.5, 2)
    # fontScale              = 1
    lineType               = np.random.randint(1, 2)
    thickness              = np.random.randint(1, 3)
    
    # put text
    img, y_line_list, text_height = print_lines(img, font, bottomLeftCornerOfText, fontColor, fontScale, lineType, thickness)

    # add noise
    noisy_img = get_noisy_img(img, y_line_list, text_height)

    # degrade_quality
    img, noisy_img = degrade_qualities(img, noisy_img)

    # morphological operations    
    # img, noisy_img =  erode_dilate(img, noisy_img)

    # make debug image
    debug_img = get_debug_image(img, noisy_img)

    # write images
    write_images(img, noisy_img, debug_img)

    '''
    cv2.imshow('textonimage', original)
    cv2.imshow('noisy_img', noisy_img)
    cv2.waitKey()
    '''
