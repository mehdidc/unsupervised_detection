from __future__ import division
import time
from itertools import islice
import math
import random
 
import numpy as np
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import resnet101, resnet152
from rampwf.workflows.image_classifier import get_nb_minibatches
from skimage.transform import resize, rotate
import warnings
warnings.filterwarnings("ignore")


### Some helpers ##
import random
import PIL
import cv2

random.seed(1603)
np.random.seed(1603)



from torch.utils.data.sampler import *


def augment2(x):
    #x = x.astype(np.float32) / 255
    u = 0.75
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    # x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    #x = 255.0 * x
    #x[:, :, 0] -= 124
    #x[:, :, 1] -= 117
    #x[:, :, 2] -= 104

    #x *= 0.0167
    #x = np.transpose(x, (2, 0, 1))
    return x


def augment(x):
    x = x.astype(np.float32) / 255
    u = 0.75
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    # x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    x = np.uint8(255.0 * np.transpose(x, (2, 0, 1)))
    return x


## custom data transform  -----------------------------------
def tensor_to_img(img, mean=0, std=1, dtype=np.uint8):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * std + mean
    img = img.astype(dtype)
    # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img


## transform (input is numpy array, read in by cv2)
def img_to_tensor(img, mean=0, std=1.):
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    tensor = torch.from_numpy(img)  ##.float()
    return tensor



# http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
# http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img, 0)  # np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img


def randomHorizontalFlip(img, u=0.5):
    shape = img.shape
    if random.random() < u:
        img = cv2.flip(img, 1)  # np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img


def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img, random.randint(-1, 1))
    return img


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1, 0, 2)  # cv2.transpose(img)
    return img


# http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees
def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle = random.randint(1, 3) * 90
        if angle == 90:
            img = img.transpose(1, 0, 2)  # cv2.transpose(img)
            img = cv2.flip(img, 1)
            # return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img, -1)
            # return img[::-1,::-1,:]
        elif angle == 270:
            img = img.transpose(1, 0, 2)  # cv2.transpose(img)
            img = cv2.flip(img, 0)
            # return  img.transpose((1,0, 2))[::-1,:,:]
    return img


def randomRotate(img, u=0.25, limit=90):
    if random.random() < u:
        angle = random.uniform(-limit, limit)  # degree

        height, width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, mat, (height, width), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    return img


def randomShift(img, u=0.25, limit=4):
    if random.random() < u:
        dx = round(random.uniform(-limit, limit))  # pixel
        dy = round(random.uniform(-limit, limit))  # pixel

        height, width, channel = img.shape
        img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1, borderType=cv2.BORDER_REFLECT_101)
        y1 = limit + 1 + dy
        y2 = y1 + height
        x1 = limit + 1 + dx
        x2 = x1 + width
        img = img1[y1:y2, x1:x2, :]

    return img


def randomShiftScale(img, u=0.25, limit=4):
    if random.random() < u:
        height, width, channel = img.shape
        assert (width == height)
        size0 = width
        size1 = width + 2 * limit
        img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
        size = round(random.uniform(size0, size1))

        dx = round(random.uniform(0, size1 - size))  # pixel
        dy = round(random.uniform(0, size1 - size))

        y1 = dy
        y2 = y1 + size
        x1 = dx
        x2 = x1 + size

        if size == size0:
            img = img1[y1:y2, x1:x2, :]
        else:
            img = cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR)

    return img


def randomShiftScaleRotate(img, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        angle = random.uniform(-rotate_limit, rotate_limit)  # degree
        scale = random.uniform(1 - scale_limit, 1 + scale_limit)
        dx = round(random.uniform(-shift_limit, shift_limit)) * width
        dy = round(random.uniform(-shift_limit, shift_limit)) * height

        cc = math.cos(angle / 180 * math.pi) * (scale)
        ss = math.sin(angle / 180 * math.pi) * (scale)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img


def cropCenter(img, height, width):
    h, w, c = img.shape
    dx = (h - height) // 2
    dy = (w - width) // 2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]

    return img


## unconverntional augmnet ################################################################################3
## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

## barrel\pincushion distortion
def randomDistort1(img, distort_limit=0.35, shift_limit=0.25, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        # debug
        # img = img.copy()
        # for x in range(0,width,10):
        #     cv2.line(img,(x,0),(x,height),(1,1,1),1)
        # for y in range(0,height,10):
        #     cv2.line(img,(0,y),(width,y),(1,1,1),1)

        k = random.uniform(-distort_limit, distort_limit) * 0.00001
        dx = random.uniform(-shift_limit, shift_limit) * width
        dy = random.uniform(-shift_limit, shift_limit) * height

        # map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
        # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        x, y = np.mgrid[0:width:1, 0:height:1]
        x = x.astype(np.float32) - width / 2 - dx
        y = y.astype(np.float32) - height / 2 - dy
        theta = np.arctan2(y, x)
        d = (x * x + y * y) ** 0.5
        r = d * (1 + k * d * d)
        map_x = r * np.cos(theta) + width / 2 + dx
        map_y = r * np.sin(theta) + height / 2 + dy

        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


# http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
## grid distortion
def randomDistort2(img, num_steps=10, distort_limit=0.2, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        x_step = width // num_steps
        xx = np.zeros(width, np.float32)
        prev = 0
        for x in range(0, width, x_step):
            start = x
            end = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * (1 + random.uniform(-distort_limit, distort_limit))

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = height // num_steps
        yy = np.zeros(height, np.float32)
        prev = 0
        for y in range(0, height, y_step):
            start = y
            end = y + y_step
            if end > width:
                end = height
                cur = height
            else:
                cur = prev + y_step * (1 + random.uniform(-distort_limit, distort_limit))

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


## blur sharpen, etc
def randomFilter(img, limit=0.5, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        alpha = limit * random.uniform(0, 1)

        ##kernel = np.ones((5,5),np.float32)/25
        kernel = np.ones((3, 3), np.float32) / 9 * 0.2

        # type = random.randint(0,1)
        # if type==0:
        #     kernel = np.ones((3,3),np.float32)/9*0.2
        # if type==1:
        #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])*0.5

        # kernel = alpha *sharp +(1-alpha)*blur
        # kernel = np.random.randn(5, 5)
        # kernel = kernel/np.sum(kernel*kernel)**0.5

        img = alpha * cv2.filter2D(img, -1, kernel) + (1 - alpha) * img
        img = np.clip(img, 0., 1.)

    return img


##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey


def randomBrightness(img, limit=0.2, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img


def randomContrast(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img


def randomSaturation(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1.0 - alpha) * gray
        img = np.clip(img, 0., 1.)

    return img

class BatchClassifier(object): 
    def __init__(self):
        self.net = Net()
 
    def fit(self, gen_builder):
        def adjust_learning_rate(optimizer, epoch,lr):            
            if epoch == 0: 
                lr= lr
            elif epoch==1:
                lr = lr
            elif  epoch==2:
                lr= lr
            elif  epoch==3:
                lr= 1e-3
            elif  epoch==4:
                lr= 1e-3
            elif  epoch==5:                
                lr= 1e-3
            elif  epoch==6: 
                lr= 1e-4
            elif  epoch==7:  
                lr= 1e-4                
            else:
                lr= 1e-4
            #lr = 0.000003
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return optimizer ,lr
                
        batch_size = 8
        nb_epochs = 8
        lr = 1e-2
        gen_train, gen_valid, nb_train, nb_valid = gen_builder.get_train_valid_generators(batch_size=batch_size, valid_ratio=0.1)
        self.net = self.net.cuda()
        net = self.net
        optimizer = optim.SGD(net.parameters(), lr=lr)
        nb_train_minibatches = get_nb_minibatches(nb_train, batch_size)
        nb_valid_minibatches = get_nb_minibatches(nb_valid, batch_size)
        criterion = nn.CrossEntropyLoss().cuda()
 
        for epoch in range(nb_epochs):
            optimizer, lr =adjust_learning_rate(optimizer,epoch,lr)
            print("using learning rate", lr)            
            t0 = time.time()
            net.train() # train mode
            nb_trained = 0
            nb_updates = 0
            train_loss = []
            train_acc = []
            for X, y in islice(gen_train, nb_train_minibatches):
                y = y.argmax(axis=1) # convert onehot to integers, pytorch require the class indices.
                if (epoch % 3 )==0 : #using original
                #if True:
                    X= _make_variable(X)                    
                else:
                    X = _make_variable_da(X)
                y = _make_variable(y)
                optimizer.zero_grad() # zero-out the gradients because they accumulate by default
                y_pred = net(X)
                loss = criterion(y_pred, y)
                loss.backward() # compute gradients
                optimizer.step() # update params
 
                # Loss and accuracy
                train_acc.extend(self._get_acc(y_pred, y))
                train_loss.append(loss.data[0])
                nb_trained += X.size(0)
                nb_updates += 1
                if nb_updates % 100 == 0 or nb_updates == nb_train_minibatches:
                    print('Epoch [{}/{}], [trained {}/{}], avg_loss: {:.4f}, avg_train_acc: {:.4f}'.format(epoch+1, nb_epochs, nb_trained, nb_train, np.mean(train_loss), np.mean(train_acc)))
 
            net.eval() # eval mode
            nb_valid = 0
            valid_acc = []
            for X, y in islice(gen_valid, nb_valid_minibatches):
                y = y.argmax(axis=1)
                X = _make_variable(X)
                y = _make_variable(y)
                y_pred = net(X)
                valid_acc.extend(self._get_acc(y_pred, y))
                nb_valid += y.size(0)
 
            delta_t = time.time() - t0
            print('Finished epoch {}'.format(epoch + 1))
            print('Time spent : {:.4f}'.format(delta_t))
            print('Train acc : {:.4f}'.format(np.mean(train_acc)))            
            print('Valid acc : {:.4f}'.format(np.mean(valid_acc)))
            print('Saving model')
            torch.save(self.net, 'model.th') 

    def _get_acc(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
        y_true = y_true.cpu().data.numpy()
        return (y_pred==y_true)
 
    def predict_proba(self, X):
        X = X.astype(np.float32)
        X = _make_variable(X) 
        y_proba = nn.Softmax()(self.net(X)).cpu().data.numpy()
        return y_proba
    
    
def _make_variable_da(X):
    #return image to initial state, image_preprocessor.py is just a dummy and dumb file... :)
    SHAPE_DEBUG=False
    
    if SHAPE_DEBUG:
        print("Entry X shape" , X.shape)
    X[:,2, :, :] *= 0.225
    X[:,2, :, :] += 0.406
    X[:,1, :, :] *= 0.224
    X[:,1, :, :] += 0.456
    X[:,0, :, :] *= 0.229
    X[:,0, :, :] += 0.485
    #X *= 255.
   
    
    X = X.transpose((0, 2, 3,1))    
    if SHAPE_DEBUG:
        print("X reshaped shape ", X.shape)
    for i in range(X.shape[0]):        
        x=X[i]        
        if SHAPE_DEBUG:
            print("Unit X shape ", x.shape)
        x = augment2(x)
        if SHAPE_DEBUG:
            print("x after aug shape ", x.shape)
        X[i]=x    
    X = X.transpose((0, 3, 1,2))
    #X /= 255.
    X[:,0, :, :] -= 0.485
    X[:,0, :, :] /= 0.229
    X[:,1, :, :] -= 0.456
    X[:,1, :, :] /= 0.224
    X[:,2, :, :] -= 0.406
    X[:,2, :, :] /= 0.225
    
    return Variable(torch.from_numpy(X)).cuda()
 
 
def _make_variable(X):
    return Variable(torch.from_numpy(X)).cuda()
 
class Net(nn.Module): 
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.model = resnet152(pretrained = True)
        #self.features =resnet152(pretrained = True).features
        self.model.fc = nn.Sequential(            
        #self.fc = nn.Sequential(    
            nn.Linear(2048, 10240),
            nn.BatchNorm1d(10240),
            nn.ReLU(True),            
            #nn.Linear(8192, 4096),
            #nn.BatchNorm1d(4096),
            #nn.ReLU(True),
            nn.Linear(10240, 403),
        )
 
    def forward(self, x):
        x = self.model(x)
        #x = self.features(x)
        #x = _flatten(x)
        #x = self.fc(x)        
        return x
 
 
def _flatten(x):
    return x.view(x.size(0), -1)

