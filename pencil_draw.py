#encoding=utf8
import math
import numpy as np
from scipy import signal

import util
import image_tool


conv_size_divider = 30

def im2double(mat):
    max_p = mat.max()
    min_p = mat.min()
    dis = max_p - min_p
    return (mat - float(min_p)) / dis


def grad_image(mat):
    Ix = mat[:, :-1] - mat[:, 1:]
    Iy = mat[:-1, :] - mat[1:, :]
    G = np.multiply(Ix, Ix) + np.multiply(Iy, Iy)
    G = np.sqrt(G)
    return G

def conv_kernel(kernel_size):
    assert kernel_size % 2 == 0
    #kernel_size = float(min(height, width)) / conv_size_divider
    C = np.zeros((kernel_size, kernel_size, 8))
    half = kernel_size / 2
    for n in range(8):
        if n in [0, 1, 2, 7]:
            for x in range(0, kernel_size):
                y = round((x+1 - half) * math.tanh(math.pi/8 * n))
                y = half - y
                if 0 < y and y <= kernel_size:
                    C[int(y-1), x, n] = 1
                index = (n+4) % 8
                C[:, :, index] = util.rot90_matrix(C[:, :, n])
    return C

def max_conv(C, mat, line_gama=1):
    _, _, n = C.shape
    w, h = mat.shape
    G = np.zeros((w, h, n))
    for i in xrange(n):
        G[:, :, i] = signal.convolve2d(mat, C[:, :, i], mode='same')
    Gindex = np.argmax(G, axis=2)
    #print Gindex
    Gmax = np.zeros((w, h, n))
    for i in xrange(n):
        #print (Gindex == i)
        Gmax[:,:, i] = np.multiply(mat, (Gindex == i) * 1)
    #print Gmax
    LSn = np.zeros((w, h, n))
    for i in xrange(n):
        LSn[:, :, i] = signal.convolve2d(Gmax[:,:, i], C[:, :, i], mode='same')

    LS = np.sum(LSn, axis=2)
    min_v = LS[:].min()
    max_v = LS[:].max()
    #print 'min_v', min_v, 'max_v', max_v
    LS[:] = (LS[:] - min_v)/ (max_v - min_v)
    ##line_gama越大，线条越粗
    S = (1-LS[:]) ** line_gama
    return S

def get_sketch(mat):
    C = conv_kernel(8)
    mat = im2double(mat)
    S = max_conv(C, mat)
    return S
   
def pencil_draw():
    rgb_im = image_tool.read_image("img/58.jpg")
    mat = image_tool.rgb_img2gray_matrix(rgb_im)
    S = get_sketch(mat)
    image_tool.save_image(image_tool.matrix2image(S), "S.bmp")


if __name__ == '__main__':
   pencil_draw() 
