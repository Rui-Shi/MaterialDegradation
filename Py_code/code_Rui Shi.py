import tensorflow as tf


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os
import cv2

import matplotlib.pyplot as plt
import numpy as np



from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# this function is for read image,the input is directory name
def read_directory(directory_name):
    i = 0
    n = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name,name))])
    array_of_img=np.zeros((n,300,300,3))
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test

        array_of_img[i]=cv2.imread(directory_name + "/" + filename)
        i = i+1
    return(array_of_img)

#n = len([name for name in os.listdir('D:/img/No crack') if os.path.isfile(os.path.join('D:/img/No crack',name))])
Five_cracks = read_directory("D:/img/Five cracks")
No_crack = read_directory("D:/img/No crack")
One_crack = read_directory("D:/img/One crack")
Six_crack = read_directory("D:/img/Six cracks")
Two_crack = read_directory("D:/img/Two cracks")
Three_crack = read_directory("D:/img/Three cracks")
Four_crack = read_directory("D:/img/Four cracks")


images=np.vstack((No_crack, One_crack, Two_crack, Three_crack, Four_crack,Five_cracks,Six_crack))


labels = np.hstack((np.zeros(len(No_crack)),np.zeros(len(One_crack))+1,np.zeros(len(Two_crack))+2,np.zeros(len(Three_crack))+3,
                    np.zeros(len(Four_crack))+4,np.zeros(len(Five_cracks))+5,np.zeros(len(Six_crack))+6))


k = np.random.choice([i for i in range(0,len(labels))],round(0.2*len(labels)),replace=False)
k.sort()
Nk =np.arange(0,len(labels),1)
Nk = np.delete(Nk, k)


train_images = images[Nk]
train_labels = labels[Nk]
test_images = images[k]
test_labels = labels[k]


#test_images

#array_of_img
#np.array(cv2.imread("D:\\img\\No crack\\UF1 100 30.jpg")).shape


#DIR = 'D:/img/No crack'
#print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))


#array_of_img=np.zeros((100,1460,1936,3))
#array_of_img[0]=cv2.imread("D:/img/No crack/UF1 100 30.jpg")


#new_array = np.zeros(len(No_crack))+1
            #np.twos(len(One_crack))



##Traditional neural networks
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(300, 300, 3)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(7)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)



## CNN model
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(300, 300, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(7)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
