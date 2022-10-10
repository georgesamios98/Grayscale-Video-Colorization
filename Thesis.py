import numpy as np
import pickle
from skimage import color
from glob import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy

def read_img(file, size = (256,256)):
    '''
    reads the images and transforms them to the desired size
    '''
    img = image.load_img(file, target_size=size)
    img = image.img_to_array(img)
    return img

def convert_img_size(file_paths):
    '''
    converts all images to 256x256x3
    '''
    all_images_to_array = np.zeros((len(file_paths), 256, 256, 3), dtype='int64')
    for ind, i in enumerate(file_paths):
        img = read_img(i)
        all_images_to_array[ind] = img.astype('int64')
    print('All Images shape: {} size: {:,}'.format(all_images_to_array.shape, all_images_to_array.size))
    return all_images_to_array

file_paths = glob('./images/*.jpg')
X_train = convert_img_size(file_paths)

def rgb_to_lab(img,flagL=False,flagAB=False):
    """
    Takes in RGB channels in range 0-255 and outputs L or AB channels in range -1 to 1
    """
    img = img / 255
    if flagL==True:
        l = color.rgb2lab(img)[:,:,0]
        l = l / 50 - 1
        l = l[...,np.newaxis]
        return l
    if flagAB==True:
        ab = color.rgb2lab(img)[:,:,1:]
        ab = (ab + 128) / 255 * 2 - 1
        return ab

def lab_to_rgb(img):
    """
    Takes in LAB channels in range -1 to 1 and out puts RGB chanels in range 0-255
    """
    new_img = np.zeros((256,256,3))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix = img[i,j]
            new_img[i,j] = [(pix[0] + 1) * 50,(pix[1] +1) / 2 * 255 - 128,(pix[2] +1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img

L = np.array([rgb_to_lab(image,flagL=True,flagAB=False)for image in X_train])
print(L.shape)
print(L.shape)
AB = np.array([rgb_to_lab(image,flagL=False,flagAB=True)for image in X_train])
L_AB_channels = (L,AB)

with open('l_ab_channels.p','wb') as f:
        pickle.dump(L_AB_channels,f)

def load_images(filepath):
    '''
    Loads in pickle files, specifically the L and AB channels
    '''
    with open(filepath, 'rb') as f:
        return pickle.load(f)

X_train_L, X_train_AB = load_images('l_ab_channels.p')

#Run this if you don't want to augment your data, it will be the final X_trains and X_tests
X_test_L = deepcopy(X_train_L[:320])
X_test_AB = deepcopy(X_train_AB[:320])
X_train_L = X_train_L[0:]
X_train_AB = X_train_AB[0:]
d_image_shape = (256,256,2)
g_image_shape = (256,256,1)

def generator():
        '''
        Creates the generator in Keras
        '''
        model = Sequential()
        model.add(Conv2D(64,(3,3),padding='same',strides=2, input_shape=g_image_shape)) #dont need pooling since stride=2 downsizes
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (3,3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(256, (3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(512,(3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2DTranspose(256,(3,3), strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(32,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2,(3,3),padding='same'))
        model.add(Activation('tanh'))
        l_channel = Input(shape=g_image_shape)
        image = model(l_channel)
        return Model(l_channel,image)

def discriminator():
        '''
        creates a discriminator in keras
        '''
        model = Sequential()
        model.add(Conv2D(32,(3,3), padding='same',strides=2,input_shape=d_image_shape))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,(3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128,(3,3), padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256,(3,3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        image = Input(shape=d_image_shape)
        validity = model(image)
        return Model(image,validity)

#Build the Discriminator
discriminator = discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.00008,beta_1=0.5,beta_2=0.999),
                    metrics=['accuracy'])
#Making the Discriminator untrainable so that the generator can learn from fixed gradient
discriminator.trainable = False
# Build the Generator
generator = generator()
generator.summary()
#Defining the combined model of the Generator and the Discriminator
l_channel = Input(shape=g_image_shape)
image = generator(l_channel)
valid = discriminator(image)
combined_network = Model(l_channel, valid)
combined_network.compile(loss='binary_crossentropy',
                         optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.999))
#creates lists to log the losses and accuracy
gen_losses = []
disc_real_losses = []
disc_fake_losses=[]
disc_acc = []

n = 320
y_train_fake = np.zeros([160,1])
y_train_real = np.ones([160,1])
y_gen = np.ones([n,1])



#Pick batch size and number of epochs, number of epochs depends on the number of photos per epoch set above
num_epochs=1501
batch_size=32

#run and train until photos meet expectations (stop & restart model with tweaks if loss goes to 0 in discriminator)
print(1)
for epoch in range(1,num_epochs+1):
    print("eimai sthn epoxh:",epoch)
    np.random.shuffle(X_train_L)
    l = X_train_L[:n]
    np.random.shuffle(X_train_AB)
    ab = X_train_AB[:160]
    fake_images = generator.predict(l[:160], verbose=1)

    d_loss_real = discriminator.fit(x=ab, y= y_train_real,batch_size=32,epochs=1,verbose=1)
    disc_real_losses.append(d_loss_real.history['loss'][-1])

    d_loss_fake = discriminator.fit(x=fake_images,y=y_train_fake,batch_size=32,epochs=1,verbose=1)
    disc_fake_losses.append(d_loss_fake.history['loss'][-1])
    disc_acc.append(d_loss_fake.history['acc'][-1])
    g_loss = combined_network.fit(x=l, y=y_gen,batch_size=32,epochs=1,verbose=1)
    gen_losses.append(g_loss.history['loss'][-1])

    #every 20 epochs it prints a generated photo and saves the model under that epoch
    if epoch>1490:
        print('Reached epoch:',epoch)
        for i in range(len(X_test_L)):
            pred = generator.predict(X_test_L[i].reshape(1,256,256,1))
            img = lab_to_rgb(np.dstack((X_test_L[i],pred.reshape(256,256,2))))
            plt.imshow(img)
            plt.axis('off')
            filename = ('Predicted_image%03depoch%03d.png' %(i,epoch))
            plt.savefig(filename)
            im1 = Image.open
        if epoch % 20 == 0:
              generator.save('generator_' + str(epoch)+ '_v3.h5')