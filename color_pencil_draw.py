#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
from PIL import Image
import cv2


from pencil_draw import get_S, get_T
import image_tool

def color_draw(path, outpath, gammaS=2.5, gammaI=3.5):
    im = Image.open(path)
    if im.mode == 'RGB':
        ycbcr = im.convert('YCbCr')
        Iruv = np.ndarray((im.size[1], im.size[0], 3), 'u1', ycbcr.tobytes())
        type = "colour"
    else:
        Iruv = np.array(im)
        type = "black"
    print 'gen S'
    S = get_S(Iruv[:, :, 0], gammaS)
    print 'gen T'
    T = get_T(Iruv[:, :, 0], type, gammaI)
    print 'gen R'
    ## element_wise production
    Ypencil = np.multiply(S, T)

    new_Iruv = Iruv.copy()
    new_Iruv.flags.writeable = True
    new_Iruv[:, :, 0] = Ypencil * 255

    R = cv2.cvtColor(new_Iruv, cv2.COLOR_YCR_CB2BGR)
    img = Image.fromarray(R)
    #img.show()

    #image_tool.save_image(image_tool.matrix2image(S), "46_S.jpg")
    #image_tool.save_image(image_tool.matrix2image(T), "46_T.jpg")
    img.save(outpath)


if __name__ == "__main__":
    import time
    start = time.time()
    args = sys.argv
    length = len(args)
    if length > 1:
        filename=args[1]
    else:
        filename="huangjin_8.jpeg"
    color_draw("me/"+filename, "output/"+filename)
    print 'time consumes: {}'.format(time.time() - start)
