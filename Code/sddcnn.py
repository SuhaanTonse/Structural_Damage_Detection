from tkinter import *
from tkinter import ttk
from tkinter import filedialog


import cv2
from PIL import ImageTk, Image

import numpy as np
import glob
import os

import pickle

from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Activation,SpatialDropout2D
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image


import matplotlib.pyplot as plt


def Upload_Dataset():
    global sourcePath
    sourcePath = filedialog.askdirectory()
    print(sourcePath)
            
def Train_CNN():
    global sourcePath
    #dir_path = r'C:\Users\HP\Documents\Project_Crackdetection\traindata'
    dir_path = sourcePath
    img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    print(len(img_list))


    train=ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             validation_split=0.1,
                             rescale=1./255,
                             shear_range = 0.1,
                             zoom_range = 0.1,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,)

    test=ImageDataGenerator(rescale=1/255,
                            validation_split=0.1)

    train_generator=train.flow_from_directory(dir_path,
                                              target_size=(300,300),
                                              batch_size=32,
                                              class_mode='categorical',
                                              subset='training')

    test_generator=test.flow_from_directory(dir_path,
                                            target_size=(300,300),
                                            batch_size=32,
                                            class_mode='categorical',
                                            subset='validation')

    labels = (train_generator.class_indices)
    print(labels)

    labels = dict((v,k) for k,v in labels.items())
    print(labels)

    for image_batch, label_batch in train_generator:
      break
    image_batch.shape, label_batch.shape


    print (train_generator.class_indices)

    Labels = '\n'.join(sorted(train_generator.class_indices.keys()))

    with open('labels.txt', 'w') as f:
      f.write(Labels)


    model=Sequential()

    #Convolution blocks
    model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    #Classification layers
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))

    filepath="trained_model.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'])
    history = model.fit(train_generator,epochs=100,steps_per_epoch=2276//32,validation_data=test_generator,validation_steps=251//32,workers = 4,callbacks=callbacks_list)
    model.save(filepath)

    f = open('history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()


def Input_Image():
    
    global file_path
    global resized_image
    root.filename =  filedialog.askopenfilename(parent=root,initialdir="/",title="choose your image",filetypes=(("png files","*.jpg"),("all files","*.*")))
    file_path = root.filename 
          
    img=cv2.imread(file_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resized_image=cv2.resize(img,(400,300))
    image=Image.fromarray(resized_image)
    image=ImageTk.PhotoImage(image)
    panelA=Label(image=image)
    panelA.image=image
    panelA.place(x=350,y=140)

def preprocessing_image():
    global resized_image
    global thresh
    gray= cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    image = Image.fromarray(thresh)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=820 ,y=140)

def w2d():
    global resized_image
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray,'haar',level=1)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H,'haar');
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    
    image = Image.fromarray(imArray_H)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=850 ,y=140)

def image_segmentation_kmeans():
    global resized_image
    global thresh
    global res_red
    hsv = cv2.cvtColor(resized_image,cv2.COLOR_BGR2HSV)
    Z = hsv.reshape((-1,3)) 
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    khsv   = center[label.flatten()]
    khsv   = khsv.reshape((resized_image.shape))
    res_red = cv2.bitwise_and(resized_image,resized_image, mask=thresh)
 
    image = Image.fromarray(res_red)
    image = ImageTk.PhotoImage(image)
    panelA = Label(image=image)
    panelA.image = image
    panelA.place(x=820 ,y=140)



def CNN_Recognition():
    print("CNN")
    global file_path
    
    dir_path = r'C:\Users\HP\Documents\Project_Crackdetection\traindata'
    img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    train=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,validation_split=0.1,rescale=1./255,shear_range = 0.1,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,)
    train_generator=train.flow_from_directory(dir_path,target_size=(300,300),batch_size=32,class_mode='categorical',subset='training')
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    model = load_model('trained_model.h5')
    
    img = keras.utils.load_img(file_path, target_size=(300, 300))
    img = keras.utils.img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0
    p=model.predict(img[np.newaxis, ...])
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:",predicted_class)
    e1.delete(0,'end')
    e1.insert(5,predicted_class)


def CNN_Plot_Graph():
    f = open('history.pckl', 'rb')
    history = pickle.load(f)
    f.close()
    print(history)
    gr=comboa3.get()
    if(gr=="ACCURACY"):
        plt.figure(0)
        plt.plot(history['accuracy'], label='training accuracy')
        plt.plot(history['val_accuracy'], label='val accuracy')
        plt.title('ACCURACY')
        plt.xlabel('EPOCHS')
        plt.ylabel('ACCURACY')
        plt.legend()
        plt.show()
    if(gr=="SPECIFICITY"):
        plt.figure(0)
        plt.plot(history['specificity_at_sensitivity'],label='SPECIFICITY')
        plt.title('SPECIFICITY')
        plt.xlabel('EPOCHS')
        plt.ylabel('SPECIFICITY')
        plt.legend()
        plt.show()
    if(gr=="SENSITIVITY"):
        plt.figure(0)
        plt.plot(history['sensitivity_at_specificity'], label='SENSITIVITY')
        plt.title('SENSITIVITY')
        plt.xlabel('EPOCHS')
        plt.ylabel('SENSITIVITY')
        plt.legend()
        plt.show()
    if(gr=="F1SCORE"):
        plt.figure(0)
        plt.plot(history['accuracy'], label='F1SCORE')
        plt.title('F1SCORE')
        plt.xlabel('EPOCHS')
        plt.ylabel('F1SCORE')
        plt.legend()
        plt.show()


def Plot_Graph():
    gr=comboa2.get()
    if(gr=="CNN"):
        CNN_Plot_Graph()
        print("CNN")

    
root = Tk() 
root.title('STRUCTURAL DAMAGE DETECTION')
root.geometry('1920x1080')
root.configure(background='gray')

c1 = Canvas(root,bg='gray',width=1266,height=80)
c1.place(x=5,y=5)
l1=Label(root,text='STRUCTURAL DAMAGE DETECTION',foreground="white",background='gray',font =('Verdana',16,'bold'))
l1.place(x=480,y=30)

c2 = Canvas(root,bg='gray',width=290,height=550)
c2.place(x=5,y=90)

l2=Label(root,text='TRAIN PHASE',foreground="white",background='gray',font =('Verdana',13,'bold'))
l2.place(x=90,y=100)

b0=Button(root,borderwidth=1,relief="flat",text="UPLOAD DATASET",font="verdana 12 bold",bg="lightgray", fg="red",command = Upload_Dataset)
b0.place(height=70,width=260,x=22,y=130)

b1=Button(root,borderwidth=1,relief="flat",text="TRAIN DATASET",font="verdana 12 bold",bg="lightgray", fg="red",command = Train_CNN)
b1.place(height=70,width=260,x=22,y=215)

l4=Label(root,text='TEST PHASE',foreground="white",background='gray',font =('Verdana',13,'bold'))
l4.place(x=95,y=355)

b2=Button(root,borderwidth=1,relief="flat",text="INPUT IMAGE", font="verdana 12 bold", bg="lightgray", fg="red",command = Input_Image)
b2.place(height=70,width=260,x=22,y=385)

b3=Button(root,borderwidth=1,relief="flat",text="PREPROCESSING",font="verdana 12 bold", bg="lightgray", fg="red",command = preprocessing_image)
b3.place(height=70,width=260,x=22,y=470)

b4=Button(root,borderwidth=1,relief="flat",text="IMAGE SEGMENTATION",font="verdana 12 bold", bg="lightgray", fg="red",command = image_segmentation_kmeans)
b4.place(height=70,width=260,x=22,y=555)

c3 = Canvas(root,bg='gray',width=972,height=400)
c3.place(x=300 ,y=90)

c4 = Canvas(root,bg='white',width=400,height=300) 
c4.place(x=350,y=140) 

c5 = Canvas(root,bg='white',width=400,height=300) 
c5.place(x=820,y=140)

c6 = Canvas(root,bg='gray',width=1057,height=145)
c6.place(x=300,y=495)

b6=Button(root,borderwidth=1, relief="flat",text="RECOGNITION",font="verdana 12 bold",bg="lightgray",fg="red",command =CNN_Recognition)
b6.place(height=70,width=260,x=350,y=555)

l6=Label(root,text='DAAMGE RECOGNIZED',foreground="white",background='gray',font =('Verdana',11,'bold'))
l6.place(x=650,y=510)
e1=Entry(root,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e1.place(height=70,width=200,x=650,y=555)

l7=Label(root,text='SELECT  GRAPH',foreground="white",background='gray',font =('Verdana',11,'bold'))
l7.place(x=900,y=510)
n3 = StringVar()
comboa3 = ttk.Combobox(root,font =('Verdana', 12,'bold'),foreground = 'red',state='readonly',textvariable = n3)
comboa3['values'] = ('ACCURACY','SPECIFICITY','SENSITIVITY','F1SCORE') 
comboa3.current(0) 
comboa3.place(height=50,width=150,x=900,y=555)

b7=Button(root,borderwidth=1, relief="flat",text="PLOT GRAPH", font="verdana 12 bold",bg="lightgray",fg="red",command = CNN_Plot_Graph)
b7.place(height=70,width=170,x=1080,y=555)

mainloop()
