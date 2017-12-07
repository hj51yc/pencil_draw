#encoding=utf8

import numpy as np


def trans_matrix(mat):
    return mat.transpose()

def rot90_matrix(mat):
    return np.asmatrix(rot90_array(np.asarray(mat)))

def rot90c_matrix(mat):
    return np.asmatrix(rot90c_array(np.asarray(mat)))

def trans_array(mat):
    """
    mat: np.array 2d
    trick 转置
    """
    return map(list, zip(*mat[::]))

def rot90_array(mat):
    """
    顺时针旋转90度
    """
    return trans_array(mat[::-1])

def rot90c_array(mat):
    """
    逆时针旋转90度
    """
    mat = mat
    for i in xrange(3):
        mat = trans_array(mat[::-1])
    return mat


if __name__ == '__main__':
    a = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
    print 'origin'
    print a
    print trans_matrix(a)
    a = np.asarray(a)
    print np.asmatrix(trans_array(a))
    print np.asmatrix(rot90_array(a))
    print np.asmatrix(rot90c_array(a))
