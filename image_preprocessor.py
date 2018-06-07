from skimage.transform import resize


def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    h, w = x.shape[:2]
    x = x[h//6: int(h-h//6), w//6: int(w-w//6), :]
    x = resize(x, (224, 224), preserve_range=True)
    x = x.transpose((2, 0, 1))
    x /= 255.
    x[0, :, :] -= 0.485
    x[0, :, :] /= 0.229
    x[1, :, :] -= 0.456
    x[1, :, :] /= 0.224
    x[2, :, :] -= 0.406
    x[2, :, :] /= 0.225
    return x


def inverse_transform(x):
    x = x.copy()
    x[0, :, :] *= 0.229
    x[0, :, :] += 0.485

    x[1, :, :] *= 0.224
    x[1, :, :] += 0.456

    x[2, :, :] *= 0.225
    x[2, :, :] += 0.406
    x = x.transpose((1, 2, 0))
    return x
