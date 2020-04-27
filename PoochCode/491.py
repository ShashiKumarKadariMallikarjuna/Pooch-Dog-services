!pip install tflearn
!pip uninstall tensorflow_estimator
!pip install tensorflow_estimator
!pip install tensorflow==1.13.2

!git clone https://shashikumarkm@bitbucket.org/shashikumarkm/dogsvscats.git

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. 

TRAIN_DIR = 'dogsvscats/train'
TEST_DIR = 'dogsvscats/test'
IMG_SIZE = 50     #50 x 50 images.. perfect square images are easier to process
LR = 1e-3      #learning rate

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

#Helper function: Process the images and convert it into a 2D array, greyscale data
def label_img(img):
    word_label = img.split('.')[-3] #Example file name is dog.92.png then it considers only dog
    # conversion to one-hot array [cat,dog]
    if word_label == 'cat': return [1,0]    #[much cat, no dog]
    elif word_label == 'dog': return [0,1]      #[no cat, very dog]

#Helper function: which helps to create a list of the training data with the image and its respective label
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #read the full path of the image
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #resize just in case, some image isnt perfect 50x50
        training_data.append([np.array(img),np.array(label)]) #this creates a list with the image (its full path) and also the label - dog or cat
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy',allow_pickle=True)

import matplotlib.pyplot as plt
from PIL import Image
for i in range(20):
  plt.subplot(4,5,i+1)
  if(train_data[i][1][0]==0):
    plt.title("DOG")
  else:
    plt.title("CAT")
  #plt.xlabel(train_data[i][1][0])
  plt.imshow(train_data[i][0], cmap=plt.get_cmap('gray'))
plt.show()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input') #50x50 is the image size

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 512, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)



convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax') #2 examples - it is either a dog or a cat
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


#model = tflearn.DNN(convnet, tensorboard_dir='log')
model = tflearn.DNN(convnet,checkpoint_path = '/tmp/tflearn_logs/',max_checkpoints=1, tensorboard_verbose=0)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500] #Training data is all images, except the last 500 images - which we are going to use as validation set
test = train_data[-500:] #This is the validation / label data, to check accuracy

#This is what is getting fit in the model
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1) #train contains image and label both
Y = [i[1] for i in train]

# this is for testing the accuracy of the model / the validation set
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

import matplotlib.pyplot as plt

def summarize_diagonistics(history):
  plt.subplot(211)
  plt.title('Cross Entropy Loss')
  plt.plot(history.history['loss'], color='blue', label='train')
  plt.plot(history.history['val_loss'], color='orange', label='test')
  # plot accuracy
  plt.subplot(212)
  plt.title('Classification Accuracy')
  plt.plot(history.history['accuracy'], color='blue', label='train')
  plt.plot(history.history['val_accuracy'], color='orange', label='test')
  # save plot to file
  filename = sys.argv[0].split('/')[-1]
  plt.savefig(filename + '_plot.png')
  plt.close()

history = model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir /tmp/tflearn_logs

model.save(MODEL_NAME)

import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[0:12]): #Just plotted the first 12 data, to see if the model predicts the right labels
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1) #3x4 sub plot for the 12 images
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0] #prediction takes a list, so it only returns the 0th element
    
    if np.argmax(model_out) == 1: str_label='Dog' #printing labels below the images
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label) 
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

