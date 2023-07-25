import cv2
import deeplake #nepotrebno ali moze biti korisno
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f* y_pred_f)
    val = (2. * intersection + K.epsilon()) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + K.epsilon())
    return val

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true,y_pred)

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #usual unet conv block sa batchnorm
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


# U-net model
def getModel(inputImage, numFilters=16, droupouts=0.1, doBatchNorm=True): #srapped model
    #encoder
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)

    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)

    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)

    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)

    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    #decoder
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)

    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)

    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model


def predict(valMap, model):
    ## util func za predict
    img = valMap['img']
    mask = valMap['mask']
    imgProc = np.array(img)
    predictions = model.predict(imgProc) 
    return predictions, imgProc, mask
    

def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(7,7))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted Mask')
    
    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('actual Mask')
    plt.show()


def toTensors(imgs, masks):
    ds = {'img': [], 'mask': []}
    for i, path in enumerate(imgs): # same len za slike i maske
        img = Image.open(path)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0  # Normalizacija
        mask = Image.open(masks[i])
        mask = mask.resize((256, 256))
        mask = np.array(mask) / 255.0  # Normalizacija
        ds['img'].append(img)
        ds['mask'].append(mask)
    print(np.shape(ds['img']), np.shape(ds['mask']))
    return ds
    
def main():
    #data = deeplake.load("hub://activeloop/drive-test")
    #dataloader = data.tensorflow()
    #print(data,dataloader)
    DIR = "DRIVE"
    TRAIN_DIR = f"{DIR}\\training\\images"
    TEST_DIR = f"{DIR}\\test\\images"
    TRAIN_MASK_DIR = f"{DIR}\\training\\mask"
    TEST_MASK_DIR = f"{DIR}\\test\\mask"
    
    X_train_imgs = [os.path.join(TRAIN_DIR, x) for x in sorted(os.listdir(TRAIN_DIR))]
    y_train_imgs = [os.path.join(TRAIN_MASK_DIR, x) for x in sorted(os.listdir(TRAIN_MASK_DIR))]
    X_test_imgs = [os.path.join(TEST_DIR, x) for x in sorted(os.listdir(TEST_DIR))]
    y_test_imgs = [os.path.join(TEST_MASK_DIR, x) for x in sorted(os.listdir(TEST_MASK_DIR))]
    
    train = toTensors(X_train_imgs,y_train_imgs)
    test = toTensors(X_test_imgs,y_test_imgs)
    
    inputs = tf.keras.layers.Input((256,256, 3))
    model = getModel(inputs, droupouts=0.07)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.fit(np.array(train['img']), np.array(train['mask']), epochs=150)
    acc = model.evaluate(np.array(test['img']),np.array(test['mask']))
    print(acc)
    preds, actuals, masks = predict(test, model)
    Plotter(actuals[1], preds[1], masks[1])
if __name__=='__main__':
    main()