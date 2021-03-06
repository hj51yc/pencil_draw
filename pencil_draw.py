#encoding=utf8
import math
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import interpolation
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve
from PIL import Image

import util
import image_tool
from stitch_function import horizontal_stitch, vertical_stitch

def im2double(I):
    Min = I.min()
    Max = I.max()
    dis = float(Max - Min)
    m, n = I.shape
    J = np.zeros((m, n), dtype="float")
    for x in range(m):
        for y in range(n):
            a = I[x, y]
            if a != 255 and a != 0:
                b = float((I[x, y] - Min) / dis)
                J[x, y] = b
            J[x, y] = float((I[x, y] - Min) / dis)
    return J


def grad_image(mat):
    h, w = mat.shape
    Ix = np.column_stack((mat[:, :-1] - mat[:, 1:], np.zeros((h, 1))))
    Iy = np.row_stack((mat[:-1, :] - mat[1:, :], np.zeros((1, w))))
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
    #Gindex = np.argmax(G, axis=2)
    Gindex = G.argmax(axis=2)
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
    S = (1-LS) ** line_gama
    return S

def get_S(mat, gamaS=1):
    """
    get sketch S of image
    """
    line_len = 10 ##建筑可以长一些，花树木小一点
    C = conv_kernel(line_len)
    mat = im2double(mat)
    grad_im = grad_image(mat)
    S = max_conv(C, grad_im, gamaS)
    return S
  
def get_Dx_Dy(hsize, wsize):
    """
    计算(某个矩阵A->向量a)后，稀疏梯度算子矩阵Dx, Dy与该向量a运算
    a * Dx = to_vec(A_dx)
    Dy * a = to_vec(A_dy)
    """
    size = hsize * wsize
    d = np.zeros((2, size))
    d[1, :] = -1 # or d[0, :] = -1

    Dx = spdiags(d, np.array([0, hsize]), size, size)
    Dy = spdiags(d, np.array([0, 1]), size, size)
    return Dx, Dy

def get_T(mat, type="black", gamaI=1, texture_file="texture.jpg", L2=0.2):
    """
    """
    print 'start histogram matching....'
    adjustI = image_tool.pencil_histogram_matching(mat, type)
    adjustI = pow(adjustI, gamaI)
    #image_tool.save_image(image_tool.matrix2image(adjustI), "46adjust.bmp")
    
    print 'start gen texture....'
    texture_img = Image.open(texture_file)
    texture = np.matrix(texture_img.convert("L"))
    texture = texture[99: texture.shape[0]-100, 99: texture.shape[1]-100]
    #texture = texture[:1024, :1024]
    texture_resize_ratio = 0.35
    ratio = texture_resize_ratio * min(mat.shape) / 1024.0
    ## 长宽都按照比例缩放
    texture_resize = interpolation.zoom(texture, (ratio, ratio))
    texture = im2double(texture_resize)
    htexture = horizontal_stitch(texture, mat.shape[1])
    H = vertical_stitch(htexture, mat.shape[0])
    #image_tool.save_image(image_tool.matrix2image(H), "46H.bmp")

    print 'start gen Dx, Dy....'
    vec_len = mat.shape[0] * mat.shape[1]
    Dx, Dy = get_Dx_Dy(mat.shape[0], mat.shape[1])
   
    print 'start to gen log_H'
    texture_1D = H.reshape((1, H.size))
    ## 转换为对角矩阵
    log_H = spdiags(np.log(texture_1D + 0.01), 0, vec_len, vec_len) ## plus 0.001 for overflow
    
    print 'start to gen log_J'
    adjustI_1D = adjustI.reshape((adjustI.size, 1))
    log_J = np.log(adjustI_1D + 0.01)
    
    print 'start to calc A, b'
    ##equation:  A * beta = b
    b = log_H.T.dot(log_J)  ## 这里因为H已经是diag, 所以transpose可用可不用
    A = log_H.T.dot(log_H) + L2 * (Dx.T.dot(Dx) + Dy.T.dot(Dy))
    
    ## calc beta 
    print 'start to spsolve'
    ## either method is ok
    #beta_1D = spsolve(A, b)
    beta_1D, info = scipy.sparse.linalg.cg(A, b, tol=1e-5, maxiter=60)

    beta = np.reshape(beta_1D, (mat.shape[0], mat.shape[1]))
    beta = (beta - beta.min()) / (beta.max() - beta.min()) * 5
    #image_tool.save_image(image_tool.matrix2image(beta), "beta.bmp")
    
    print 'start to calc T..'
    T = pow(H, beta)
    #T = (T - T.min()) / (T.max() - T.min())
    return T

def pencil_draw(path, outpath):
    #rgb_im = image_tool.read_image("img/58.jpg")
    #mat = image_tool.rgb_img2gray_matrix(rgb_im)
    #im = Image.open("me/huangjin_1.jpg")
    im = Image.open(path)
    im = im.convert("L")
    mat = np.array(im)
    print 'gen S'
    S = get_S(mat, 2)
    #image_tool.save_image(image_tool.matrix2image(S), "debug_S.bmp")
    
    print 'gen T'
    T = get_T(mat, 'black', 3.5)
    #image_tool.save_image(image_tool.matrix2image(T), "debug_T.bmp")
    
    print 'gen R'
    #R = S * T
    R = np.multiply(S, T)
    image_tool.save_image(image_tool.matrix2image(R), outpath)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "huangjin_8.jpeg"
    pencil_draw("me/"+filename, "output/"+filename)
