import numpy as np
from scipy.misc import imresize

def resize(img, hs, ws):
    return imresize(img, size=(hs, ws))

def deprocess(img, w = None, h = None, typ = 'float64'):
        
    if not w or not h:
        ims = img.shape
        w = ims[1]
        h = ims[2]
    
    R_CH = img[0]
    G_CH = img[1]
    B_CH = img[2]
    
    iimg = np.zeros((w, h, 3), dtype=typ)
    
    iimg[:,:,0] = R_CH
    iimg[:,:,1] = G_CH
    iimg[:,:,2] = B_CH
    
    return iimg

def preprocess(img, w, h, typ = 'float64'):
    
    if img.shape[0] != w or img.shape[1] != h:
        img = resize(img, w, h)
        
    img = img.astype(typ)
    
    R_CH = img[:,:,0]
    G_CH = img[:,:,1]
    B_CH = img[:,:,2]

    itenzor = np.zeros((1,3, w, h))

    itenzor[0,0] = B_CH
    itenzor[0,1] = G_CH
    itenzor[0,2] = R_CH
    
    return itenzor

def flint(i):
    return int(np.floor(i))

def crop(im, ri):
    xl, yl, xh, yh = ri[1:5]
    xl, yl, xh, yh = flint(xl), flint(yl), flint(xh), flint(yh)
    
    xs = flint(xh-xl)
    ys = flint(yh-yl)
    dat = np.zeros((3, ys, xs))
    dat[0] = im[0][yl:yh,xl:xh]
    dat[1] = im[1][yl:yh,xl:xh]
    dat[2] = im[2][yl:yh,xl:xh]
    
    return dat

def roi_pool(d, r):
    stack = []
    s = 227
    for RoI in r:
        di = int(RoI[0])
        stack.append(preprocess(resize(deprocess(crop(d[di],RoI), typ='int32'), s, s), s, s, typ='int32'))

    inp = np.vstack(stack)
    return inp
