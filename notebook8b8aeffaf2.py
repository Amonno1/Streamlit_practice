# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="l3lrsXb4MOh2"
# ## In this tutorial, I will show how to code a license plate recognizer for Indian license plates using deep learning and some image processing.
# ### Find the detailed explanation of the project in this blog: https://towardsdatascience.com/ai-based-indian-license-plate-detector-de9d48ca8951?source=friends_link&sk=a2cbd70e630f6dc3d030e3bae34d98ef

# + executionInfo={"elapsed": 5199, "status": "ok", "timestamp": 1675349277011, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="ycRjhI25UC-P"
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import streamlit as st
st.markdown("#")
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23011, "status": "ok", "timestamp": 1675349528899, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="iUPg0_x3MjII" outputId="2c0e560c-9079-4656-b654-85c7549cd8a6"
st.header("Nhận diện biển số xe")
# + executionInfo={"elapsed": 1225, "status": "ok", "timestamp": 1675349532690, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="fMDZHcIuGJLe"
# Loads the data required for detecting the license plates from cascade classifier.
plate_cascade = cv2.CascadeClassifier('license_plate.xml')
# add the path to 'india_license_plate.xml' file.

# + executionInfo={"elapsed": 368, "status": "ok", "timestamp": 1675349534912, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="r6BZ2WY8GJHM"
def detect_plate(img, text=''): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) # finally representing the detected contours by drawing rectangles around the edges.
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)
        
    return plate_img, plate # returning the processed image.


# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 1545, "status": "ok", "timestamp": 1675349539018, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="RJ8ScvVJGgH_" outputId="f53de1e6-b6d5-4a50-86bd-5ef0982dd9a9"
# Testing the above function
def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

img = cv2.imread('car3.jpg')
display(img, 'input image')
st.image(img, caption="Ảnh gốc")
# + executionInfo={"elapsed": 364, "status": "ok", "timestamp": 1675349542413, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="TIMcAMmUGgFB"
# Getting plate prom the processed image
output_img, plate = detect_plate(img)

# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 1190, "status": "ok", "timestamp": 1675349545342, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="zCfOMO__HEUf" outputId="95e09ab6-2215-42b8-f826-e74a913c1528"
display(output_img, 'detected license plate in the input image')
st.image(output_img, caption="Ảnh biển số xe được phát hiện")
# + colab={"base_uri": "https://localhost:8080/", "height": 185} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1675349547199, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="kGk622P-HERv" outputId="97563391-612a-4044-b972-0a5bad4ba5f4"
display(plate, 'extracted license plate from the image')
st.image(plate, caption="Ảnh biển số xe trích xuất từ ảnh")

# + executionInfo={"elapsed": 653, "status": "ok", "timestamp": 1675349550048, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="MzopHrMvUC-Z"
# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    st.image(ii, caption="phát hiện đường viền từ ảnh")    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


# + executionInfo={"elapsed": 443, "status": "ok", "timestamp": 1675349555130, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="h23diSmEUC-e"
# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


# + colab={"base_uri": "https://localhost:8080/", "height": 237} executionInfo={"elapsed": 761, "status": "ok", "timestamp": 1675349558407, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="OGhFmSnYUC-j" outputId="bb2af9d0-84ff-4ec6-de81-e38fb343d01f"
# Let's see the segmented characters
char = segment_characters(plate)

# + colab={"base_uri": "https://localhost:8080/", "height": 96} executionInfo={"elapsed": 436, "status": "ok", "timestamp": 1675349561365, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="rZoiyrDaUC-p" outputId="bdfcaa24-f33f-4648-b47e-0ec28c35f8aa"
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')

# + [markdown] id="QXhqHfXLUC-9"
# ### Model for characters
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8991, "status": "ok", "timestamp": 1675349575463, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="BhrsmfX9UC_p" outputId="43e6712b-8f93-40f4-e230-18c8127f864e"
import tensorflow.keras.backend as K
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data/data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
        path+'/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28 batch_size=1,
        class_mode='sparse')


# + executionInfo={"elapsed": 680, "status": "ok", "timestamp": 1675349598193, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="WXdiO1Kq9kPI"
# Metrics for checking the model performance while training
def f1score(y, y_pred):
  return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
  return tf.py_function(f1score, (y, y_pred), tf.double)
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 751, "status": "ok", "timestamp": 1675349600236, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="8IjCdBYrp4EK" outputId="e6bd8908-b8ca-4bd0-a437-c06079e19947"
K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1675349603378, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="tLCaQUxMMOid" outputId="d78f17ff-82dc-49f2-a12e-2889698343ce"
model.summary()


# + executionInfo={"elapsed": 368, "status": "ok", "timestamp": 1675349606486, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="w5aaqsHABUwx"
class stop_training_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_custom_f1score') > 0.99):
      self.model.stop_training = True


# + colab={"base_uri": "https://localhost:8080/"} id="KPAtDd_Jp4BP" outputId="016d8e43-7564-47f4-b3da-c19d61d95d77"
batch_size = 1
callbacks = [stop_training_callback()]
model.fit_generator(
      train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator, 
      epochs = 80, verbose=1, callbacks=callbacks)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1460, "status": "ok", "timestamp": 1675332727854, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="3PICNwtZUDAD" outputId="5aee10b7-69b0-401d-9e9c-b0337f373174"
# Predicting the output
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        #y_ = model.predict_classes(img)[0] #predicting the class
        y_=model.predict(img)[0]
        y=np.argmax(y_,axis=0)
        character = dic[y] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

print(show_results())

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 5746, "status": "ok", "timestamp": 1675332739510, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="urZpH4YFUDAI" outputId="fc6147fd-56d1-4a54-8ff2-ed576d4e5314"
# Segmented characters and their predicted value.
plt.figure(figsize=(10,6))
for i,ch in enumerate(char):
    img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
    plt.subplot(3,4,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted: {show_results()[i]}')
    plt.axis('off')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 496} executionInfo={"elapsed": 1540, "status": "error", "timestamp": 1675332745203, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="uBboEZgAUDAT" outputId="e1aaf907-d49d-4a62-dab1-62a57f2f120a"
#plate_number = show_results()
#output_img, plate = detect_plate(img, plate_number)
#display(output_img, 'detected license plate number in the input image')

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 677, "status": "ok", "timestamp": 1675332833776, "user": {"displayName": "Th\u1ebf Trung Tr\u1ea7n", "userId": "01672387109351568284"}, "user_tz": -420} id="_SIPpP-FMOii" outputId="ae556f3a-7054-49d5-eb70-6951e8b1cda5"
plate_number = show_results()
a=plate_number
print("detected license plate number in the input image is:",a)
st.write("Biển số xe sau khi dự đoán: ",a)

