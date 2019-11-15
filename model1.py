from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D,Input,Dense,Lambda,maximum,concatenate,SeparableConv2D,add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras

#Download dataset from https://drive.google.com/open?id=1wWuxCQJEOQX980LuwSjTBM-EzbOJQtJy

BATCH_SIZE = 3
MAX_EPOCH = 300
IMAGE_SIZE = (256,256)
TRAIN_IM = 250
VALIDATE_IM = 50



input = Input(shape =(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
conv1_1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
conv1_2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv1_1)
conv2_1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
convA = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(input)
convA1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convA)
convA2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convA1)
added1 = add([conv1_2,conv2_1 ,convA2])
pool1 = MaxPool2D(pool_size=(2,2))(added1)


conv4_1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool1)
conv4_2 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv4_1)
conv5_1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool1)
added2 = add([conv4_2,conv5_1])
pool2 = MaxPool2D(pool_size=(2,2))(added2)

conv8_1 = Conv2D(60, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool2)
conv8_2 = Conv2D(60, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv8_1)
conv9_1 = Conv2D(60, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(pool2)
added3 = add([conv8_2,conv9_1])

conv11_1 = Conv2D(60, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(added3)
conv11_2 = Conv2D(60, 2 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv11_1)
conv12_1 = Conv2D(60, 2,padding='same',activation='relu',kernel_initializer='he_normal')(added3)
added4 = add([conv11_2,conv12_1])

upsim16_1 = UpSampling2D(size=(2,2))(added4)
conv14_1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim16_1)
conv14_2 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv14_1)
conv15_1 = Conv2D(40, 4 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim16_1)
added5 = add([conv15_1,conv14_2])

upsim17_1 = UpSampling2D(size=(2,2))(added5)
conv17_1 = Conv2D(20, 8,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
conv17_2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(conv17_1)
conv18_1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
convB = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(upsim17_1)
convB1 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convB)
convB2 = Conv2D(20, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(convB1)
added6 = add([conv18_1,conv17_2,convB2])

conv20 = Conv2D(15, 8 ,padding='same',activation='relu',kernel_initializer='he_normal')(added6)
output =Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv20)
model = Model(inputs=input, outputs=output)



print(model.summary())


def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy',iou])

def myGenerator(type):
    datagen = ImageDataGenerator(rescale=1./255)

    input_generator = datagen.flow_from_directory(
        'textlocalize/'+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        'textlocalize/'+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

checkpoint = ModelCheckpoint('my_model.h5', verbose=1, monitor='val_iou',save_best_only=True, mode='max')


class ShowPredictSegment(Callback):
    def on_epoch_end(self, epoch, logs={}):
        testfileloc = ['textlocalize/validation/Input/1.jpg',
                       'textlocalize/validation/Input/2.jpg',
                       'textlocalize/validation/Input/3.jpg',
                       'textlocalize/validation/Input/4.jpg',
                       'textlocalize/validation/Input/5.jpg']

        for k in range(len(testfileloc)):
            test_im = cv2.imread(testfileloc[k])
            true_size = test_im.shape
            if true_size[1] >=  true_size[0]:
                imshow_size = (300, round(true_size[0] * 300 / true_size[1]))
            else:
                imshow_size = (round(true_size[1] * 300 / true_size[0]),300)
            cv2.imshow('Input'+str(k), cv2.resize(test_im, imshow_size))
            cv2.moveWindow('Input'+str(k), 20 + 350 * k,10)

            test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
            test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            test_im = test_im / 255.
            test_im = np.expand_dims(test_im, axis=0)
            segmented = model.predict(test_im)
            segmented = np.around(segmented)
            segmented = (segmented[0, :, :, 0] * 255).astype('uint8')
            cv2.imshow('Output'+str(k), cv2.resize(segmented, imshow_size))
            cv2.moveWindow('Output'+str(k), 20 + 350 * k,400)
            cv2.waitKey(100)

show_result = ShowPredictSegment()

h = model.fit_generator(myGenerator('Dataset'),
                        steps_per_epoch=TRAIN_IM/BATCH_SIZE,
                        epochs=MAX_EPOCH,
                        validation_data=myGenerator('validation'),
                        validation_steps=VALIDATE_IM/BATCH_SIZE,
                        callbacks=[checkpoint,show_result])

plt.plot(h.history['iou'])
plt.plot(h.history['val_iou'])
plt.show()
