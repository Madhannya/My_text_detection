from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D,Input,Dense,Lambda,Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Download dataset from https://drive.google.com/open?id=1wWuxCQJEOQX980LuwSjTBM-EzbOJQtJy

BATCH_SIZE = 5
MAX_EPOCH = 100
IMAGE_SIZE = (256,256)
TRAIN_IM = 160
VALIDATE_IM = 15

def R(x):
    return x[0]
def G(x):
    return x[1]
def B(x):
    return x[3]

####### from sensei ######
# model = Sequential()
# model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
#                  input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))  


###### without "Sequential" + 4 hidden layer #####

# input = Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
# conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
# pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
# conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
# pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
# conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
# pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
# conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
# pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
# conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
# conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
# UPSIM1 = UpSampling2D(size=(2, 2))(conv6)
# conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UPSIM1)
# UPSIM2 = UpSampling2D(size=(2, 2))(conv7)
# conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UPSIM2)
# UPSIM3 = UpSampling2D(size=(2, 2))(conv8)
# conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UPSIM3)
# UPSIM4 = UpSampling2D(size=(2, 2))(conv9)
# conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UPSIM4)
# output = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv10)
# model = Model(inputs=input, outputs=output)


### Version 1 (not finish yet)####

input = Input(shape =(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
Layer_R = Lambda(R)(input)
Layer_G = Lambda(G)(input)
Layer_B = Lambda(B)(input)
convR1 = Conv2D(16, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(Layer_R)
convG1 = Conv2D(16, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(Layer_G)
convB1 = Conv2D(16, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(Layer_B)
poolR1 = MaxPool2D(pool_size=(2, 2))(convR1)
poolG1 = MaxPool2D(pool_size=(2, 2))(convG1)
poolB1 = MaxPool2D(pool_size=(2, 2))(convB1)
convR2 = Conv2D(32, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolR1)
convG2 = Conv2D(32, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolG1)
convB2 = Conv2D(32, 3 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolB1)
poolR2 = MaxPool2D(pool_size=(3, 3))(convR2)
poolG2 = MaxPool2D(pool_size=(3, 3))(convG2)
poolB2 = MaxPool2D(pool_size=(3, 3))(convB2)
convR2 = Conv2D(64, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolR2)
convG2 = Conv2D(64, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolG2)
convB2 = Conv2D(64, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(poolB2)
concat1 = Concatenate([convR2,convG2])
concat2 = Concatenate([concat1,convB2])
model = Model(inputs=input, outputs=concat2)

######## version 2 ####







print(model.summary())


# def iou(y_true, y_pred):
#     y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
#     y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
#     inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
#     union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
#     return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

# model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy',iou])

# def myGenerator(type):
#     datagen = ImageDataGenerator(rescale=1./255)

#     input_generator = datagen.flow_from_directory(
#         'textlocalize/'+type,
#         classes = ['Input'],
#         class_mode=None,
#         color_mode='rgb',
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         seed = 1)

#     expected_output_generator = datagen.flow_from_directory(
#         'textlocalize/'+type,
#         classes = ['Output'],
#         class_mode=None,
#         color_mode='grayscale',
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         seed = 1)

#     while True:
#         in_batch = input_generator.next()
#         out_batch = expected_output_generator.next()
#         yield in_batch, out_batch

# checkpoint = ModelCheckpoint('my_model.h5', verbose=1, monitor='val_iou',save_best_only=True, mode='max')


# class ShowPredictSegment(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         testfileloc = ['textlocalize/validation/Input/001.jpg',
#                        'textlocalize/validation/Input/034.jpg',
#                        'textlocalize/validation/Input/045.jpg',
#                        'textlocalize/validation/Input/089.jpg']

#         for k in range(len(testfileloc)):
#             test_im = cv2.imread(testfileloc[k])
#             true_size = test_im.shape
#             if true_size[1] >=  true_size[0]:
#                 imshow_size = (300, round(true_size[0] * 300 / true_size[1]))
#             else:
#                 imshow_size = (round(true_size[1] * 300 / true_size[0]),300)
#             cv2.imshow('Input'+str(k), cv2.resize(test_im, imshow_size))
#             cv2.moveWindow('Input'+str(k), 20 + 350 * k,10)

#             test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
#             test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
#             test_im = test_im / 255.
#             test_im = np.expand_dims(test_im, axis=0)
#             segmented = model.predict(test_im)
#             segmented = np.around(segmented)
#             segmented = (segmented[0, :, :, 0] * 255).astype('uint8')
#             cv2.imshow('Output'+str(k), cv2.resize(segmented, imshow_size))
#             cv2.moveWindow('Output'+str(k), 20 + 350 * k,400)
#             cv2.waitKey(100)

# show_result = ShowPredictSegment()

# h = model.fit_generator(myGenerator('train'),
#                         steps_per_epoch=TRAIN_IM/BATCH_SIZE,
#                         epochs=MAX_EPOCH,
#                         validation_data=myGenerator('validation'),
#                         validation_steps=VALIDATE_IM/BATCH_SIZE,
#                         callbacks=[checkpoint,show_result])

# plt.plot(h.history['iou'])
# plt.plot(h.history['val_iou'])
# plt.show()
