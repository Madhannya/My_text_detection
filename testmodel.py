import glob
from keras.models import Model,Sequential,load_model
from keras.layers import Conv2D, MaxPool2D, UpSampling2D,Input,Dense,Lambda,maximum,concatenate,SeparableConv2D,add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras

def iou(y_true, y_pred):
    I = cv2.bitwise_and(y_true,y_pred)
    U = cv2.bitwise_or(y_true,y_pred)
    score = (np.sum(I)+0.0001)/(np.sum(U)+0.0001)
    return score

IMAGE_SIZE = (256,256)



input = Input(shape =(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
conv1_1 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
conv1_2 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv1_1)
conv2_1 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
convA = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
convA1 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convA)
convA2 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convA1)
# convC = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
# convC1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convC)
# convC2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convC1)
# convC3 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convC2)
# added1 = add([conv1_2,conv2_1 ,convA2,convC3])
added1 = add([conv1_2,conv2_1 ,convA2])
pool1 = MaxPool2D(pool_size=(2,2))(added1)


conv4_1 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool1)
conv4_2 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv4_1)
conv5_1 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool1)
# convD = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool1)
# convD1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(convD)
# convD2 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(convD1)
# added2 = add([conv4_2,conv5_1,convD2])
added2 = add([conv4_2,conv5_1])
pool2 = MaxPool2D(pool_size=(2,2))(added2)

conv8_1 = Conv2D(48, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool2)
conv8_2 = Conv2D(48, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv8_1)
conv9_1 = Conv2D(48, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool2)
added3 = add([conv8_2,conv9_1])

conv11_1 = Conv2D(48, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(added3)
conv11_2 = Conv2D(48, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv11_1)
conv12_1 = Conv2D(48, 2,padding='same',activation='relu',kernel_initializer='he_normal')(added3)
# added4 = add([conv11_2,conv12_1,added3])
added4 = add([conv11_2,conv12_1])

upsim16_1 = UpSampling2D(size=(2,2))(added4)
conv14_1 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim16_1)
conv14_2 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv14_1)
conv15_1 = Conv2D(24, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim16_1)
# convE = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim16_1)
# convE1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(convE)
# convE2 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(convE1)
# added5 = add([conv15_1,conv14_2,added2])
# added5 = add([conv15_1,conv14_2,convE2])
added5 = add([conv15_1,conv14_2])

upsim17_1 = UpSampling2D(size=(2,2))(added5)
conv17_1 = Conv2D(12, 8,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
conv17_2 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv17_1)
conv18_1 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
convB = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
convB1 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convB)
convB2 = Conv2D(12, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convB1)
# convF = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
# convF1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convF)
# convF2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convF1)
# convF3 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convF2)
# added6 = add([conv18_1,conv17_2,added1,convB2])
# added6 = add([conv18_1,conv17_2,convB2,convF3])
added6 = add([conv18_1,conv17_2,convB2])

conv20 = Conv2D(15, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(added6)
output =Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv20)
model = Model(inputs=input, outputs=output)

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy',iou])
# model = custom_unet(
#     input_shape=(256, 256, 3),
#     use_batch_norm=True,
#     num_classes=1,
#     filters=32,
#     dropout=0,
#     num_layers=4,
#     output_activation='sigmoid')

model.load_weights('my_model.h5')



path = glob.glob("fisishTest/validation/Input/*.jpg")
mean_score = list()

for myfile in path:
    test_im = cv2.imread(myfile)
    true_size = test_im.shape
    imshow_size = (512,round(true_size[0]*512/true_size[1]))
    #cv2.imshow('Input',cv2.resize(test_im, imshow_size))

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    segmented = np.around(segmented)
    im_true = cv2.imread(myfile.replace("Input","Output"),0)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')
    im_pred = cv2.resize(segmented, imshow_size)
    #cv2.imshow('Output',im_pred)
    im_pred = cv2.resize(im_pred, (im_true.shape[1],im_true.shape[0]), interpolation = cv2.INTER_AREA)
    #im_true =  cv2.resize(im_true, IMAGE_SIZE)
    #im_pred =  cv2.resize(im_pred, IMAGE_SIZE)
    myfile = myfile.replace("validation","pre_test")
    cv2.imwrite(myfile,im_pred)
    im_true = im_true
    im_pred = im_pred
    score = iou(im_true,im_pred)
    mean_score.append(score)
    print(score)
    #cv2.waitKey()

print("Total:",np.mean(mean_score))
