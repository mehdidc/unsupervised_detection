import argparse
from clize import run
import sys
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from skimage.io import imsave

# Specific to the model
from image_preprocessor import transform, inverse_transform
from batch_classifier import Net


def detect(filename, mask_threshold=0.0001, xscale=1.5, yscale=1.5, out='out.png'):
    # Load model
    model = Net()
    model.double()
    model.load_state_dict(
        torch.load('model.th', map_location=lambda storage, loc: storage))
    model.eval()

    # Load and preprocess image
    image = Image.open(filename)
    image = np.asarray(image, dtype=np.uint8)
    orig_image = image
    image = transform(image)
    X = torch.from_numpy(image)
    X = X.contiguous()
    X = X.view(1, X.size(0), X.size(1), X.size(2))
    X = Variable(X, requires_grad=True)

    # Get gradients of inputs with respect to class with max probability
    grads = {}

    def store_val(x):
        grads['dx'] = x
    X.register_hook(store_val)
    y_pred = nn.Softmax()(model(X))
    vals, indices = y_pred.max(1)
    L = y_pred[0, indices[0].data[0]]
    L.backward()

    # Compute mask
    xgrad = grads['dx']
    mask = xgrad.data.abs().max(1)[0].cpu().numpy() >= mask_threshold
    mask = mask[0]
    # Deprocess image
    image = inverse_transform(image)
    # Get bounding box
    yy, xx = np.indices(mask.shape)
    mx = xx[mask].mean()
    mx_std = xx[mask].std()
    my = yy[mask].mean()
    my_std = yy[mask].std()
    mx_std = mx_std * xscale
    my_std = my_std * yscale
    x, y, w, h = max(mx - mx_std, 0), max(my - my_std, 0), mx_std * 2, my_std * 2
    x = (x / image.shape[1]) * orig_image.shape[1]
    y = (y / image.shape[0]) * orig_image.shape[0]
    w = (w / image.shape[1]) * orig_image.shape[1]
    h = (h / image.shape[0]) * orig_image.shape[0]
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    m = np.zeros_like(orig_image)
    m[y:y+h, x:x+w] = 1
    image_masked = orig_image * m
    imsave('out.png', image_masked)
    print(x, y, w, h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="image path") 
    parser.add_argument("--mask-threshold", help='Mask threshold', default=0.0001)
    parser.add_argument("--xscale", help='scale', default=1.5)
    parser.add_argument("--yscale", help='scale', default=1.5)
    parser.add_argument("--out", help='out image path', default='out.png')
    args = parser.parse_args()
    detect(
        args.filename,
        mask_threshold=args.mask_threshold,
        xscale=args.xscale,
        yscale=args.yscale,
        out=args.out
    )
