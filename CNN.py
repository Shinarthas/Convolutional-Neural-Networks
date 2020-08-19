#importing the libraries
import tensorflow as tf
#image preprocessor
from keras.preprocessing.image import ImageDataGenerator

#Data Preprossing

#processing the Training set
# только на тренировочной для того что-бы не попасть в overfitting
# применим трансвормации зумы и попороты как в фотошопе с картинками image augmentation https://miro.medium.com/max/605/0*Utma-dS47hSoQ6Zt
train_datagen = ImageDataGenerator(
        rescale=1./255,#разделить каждый пиксель на 255
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), # размер картинки к которой нужно привести картинки которые скормим модели
        batch_size=32, # сколько картинок делать с каждой?
        class_mode='binary')

#preprocessing test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set=train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),  # размер картинки к которой нужно привести картинки которые скормим модели
        batch_size=32,  # сколько картинок делать с каждой?
        class_mode='binary')

#Building the CNN

#инициализация
cnn= tf.keras.models.Sequential()# последовательность слоев

#Добавим слой конфолюции
#filters - количество детекторов фич
#kernel_size размер матрицы детектора фич (3) - 3x3 матрица
#activation функция активации
# Relu активатов __/
# input_shape размер массив мы отправляем в модель т.е. размер картинки
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#Pulling
# добавим слой который выжмет из фич то что надо
# pool_size=размер окна по которым будем собирать инфу (2)
# strides - размер шага, есть ли оверлап или нет (2) - тогда овердепа нет
# padding инструкция что делать когда мы дошли до края картинки
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

# Добавим еще 1 точно такой же слой
# input_shape нужно убрать так ка он нужен только при инициализации
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

#Flattening
#там где мы все наши фичи обьеденияем в массив 1d
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#adding output layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Training CNN
#Compiling the CNN
#Loss такой потому что мы предсказываем бинарную переменную иначе categorical_crossentropy
#если классификация не бинарная categorical_crossentropy
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the Ann on the training set
cnn.fit(x=training_set,validation_data=test_set,epochs=1)

#making a single prediction
import numpy as np
from keras.preprocessing.image import image
# надо заресайзить для НН
test_image1=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64, 64))
test_image2=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64, 64))
test_image1=image.img_to_array(test_image1)
test_image2=image.img_to_array(test_image2)

# добавим еще 1 размерность так как до этого мы определили что для тренировочной выборки мы будем использовать серии изображений по 32
# и указываем на каком уровне расширим
test_image1=np.expand_dims(test_image1,axis=0)

result1=cnn.predict(test_image1)


#  нажо узнать 1 это кошка или собака, и 0 это собака или кошка
print('------')
print(training_set.class_indices)
print('------')
# 1- собака 0-кошка
if result1[0][0]==1:
        prediction='dog'
else:
        prediction = 'cat'

print(prediction)