from PIL import Image
import numpy as np

import util

def read_image(filename):
    return Image.open(filename)

def image2matrix(img):
    width, height = img.size
    p = img.convert("L")
    mat = np.matrix(p.getdata(), dtype='float')/255.0
    return np.reshape(mat, (height, width))

def matrix2image(mat):
    data = mat * 255
    img = Image.fromarray(data.astype(np.uint8))
    return img

def rgb_img2gray_matrix(rgb_img):
    r, g, b = rgb_img.split()
    r, g, b = image2matrix(r), image2matrix(g), image2matrix(b) 
    return r * 0.299 + g * 0.587 + b * 0.114

def rgb_img2rgb_matrix(rgb_img):
    r, g, b = rgb_img.split()
    r, g, b = image2matrix(r), image2matrix(g), image2matrix(b) 
    return r, g, b

def save_image(img, filename):
    img.save(filename)




if __name__ == "__main__":
    rgb_im = read_image("img/58.jpg")
    mat = rgb_img2gray_matrix(rgb_im)
    gray_im = matrix2image(mat)
    save_image(gray_im, "hj_test.bmp")
    mat_90 = util.rot90_matrix(mat)
    save_image(matrix2image(mat_90), "hj_test.90.bmp")
    mat_90c = util.rot90c_matrix(mat)
    save_image(matrix2image(mat_90c), "hj_test.90c.bmp")

    #mat = image2matrix(rgb_im)
    #gray_im = matrix2image(mat)
    #save_image(gray_im, "hj_test.bmp")
    
