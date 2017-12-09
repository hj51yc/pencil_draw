#encoding=utf8

from PIL import Image
import numpy as np
import math

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



def p1(v):
    """
    laplace distribution
    """
    theta_b = 9.0
    if 256 < v:
        return 0
    return math.exp((v-256)/theta_b) * (1/theta_b)

def p2(v):
    """
    uniform distribution
    """
    ua, ub = 105, 225
    if v >= ua and v <= ub:
        return 1.0/(ub - ua)
    return 0

def p3(v):
    """
    gaussian distribution
    """
    ud, theta_d = 90, 11
    a =  math.exp(-math.pow(v - ud, 2)/(2*math.pow(theta_d, 2)))
    b = math.sqrt(2 * math.pi * theta_d)
    return a / b

def tone_distribution(v, type='black'):
    """
    type: 1: black, 2: color
    """
    if type == 'black':
        return 76 * p1(v) +  22 * p2(v) + 2 * p3(v)
    else:
        return 62 * p1(v) +  30 * p2(v) + 5 * p3(v)


def pencil_histogram_matching(mat, type="black"):
    """
    mat: image with value [1 - 256]
    type: 1: black , 2: color
    
    ### 根据素描的直方图分布，做直方图匹配
    """
    ## calc cur image histogram distribution
    h, w = mat.shape
    ho = np.zeros((1, 256))
    po = np.zeros((1, 256))
    for i in xrange(256):
        po[0, i] = np.sum(1* (mat == i))
    po /= float(np.sum(po))
    ho[0, 0] = po[0, 0]
    for i in xrange(1, 256):
        ho[0, i] = ho[0, i-1] + po[0, i]
    
    ### 素描直方图匹配
    histo = np.zeros((1, 256))
    prob = np.zeros((1, 256))
    for i in xrange(256):
        prob[0, i] = tone_distribution(i, type)
    prob /= float(np.sum(prob))
    histo[0, 0] = prob[0, 0]
    for i in xrange(1, 256):
        histo[0, i] = histo[0, i-1] + prob[0, i]
   
    ## 直方图匹配
    adjustI = np.zeros((h, w))
    for i in xrange(h):
        for j in xrange(w):
            hist_value = ho[0, mat[i, j]]
            index = np.argmin(abs(histo - hist_value))
            #adjustI[i, j] = histo[0, index]
            adjustI[i, j] = index
    adjustI /= 255.0
    return adjustI
    


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
    
