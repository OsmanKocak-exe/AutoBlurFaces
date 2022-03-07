import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
from keras.preprocessing import image
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from PIL import ImageFilter

mainW = ThemedTk(theme="equilux", themebg=True, toplevel=True)
mainW.title("AI Auto Face Detect & Blur")
mainWFX = 650
mainWFY = 750
mainW.geometry(str(mainWFX)+"x"+str(mainWFY))
ttkstyle = ttk.Style()
ttkstyle.configure('Red.TLabelframe.Label',font=("TkDefaultFont", 12, "bold"))
btntrain = ttk.Button(mainW, text="APPLY BLUR", command=lambda: checkNull()).place(x=1, y=150, height=50, width=649)
btnUpload = ttk.Button(mainW, text='Select Image', command=lambda: UploadFunc()).place(x=1,y=10,width=649)
img_path = ''
label = ttk.Label(mainW, text= "")
label.place(x=1, y=125)
label2 = ttk.Label(mainW, text= "osmankocakank@gmail.com")
label2.place(x=1, y=725)
currentcmbx_var = StringVar()
currentcmbx_var.set('        Select Blur Level')
radiuSCmb = ttk.Combobox(mainW, textvariable=currentcmbx_var)
radiuSCmb.place(x=250 , y=49)
radiuSCmb['values'] = ('Blur Level: 1','Blur Level: 10','Blur Level: 25','Blur Level: 35', 'Blur Level: 50', 'Blur Level: 75', 'Blur Level: 100')
radiuSCmb['state'] = 'readonly'

def UploadFunc():
    global img_path
    filepath = filedialog.askopenfilename()
    img_path = filepath
    label.config(text="Selected image : "+ filepath)

detect_m_path = 'data/detection_models/haarcascade_frontalface_default.xml'

rgb_offsets = (30, 60)
rgb_offsets = (10, 10)
gry_offsets = (20, 40)
gry_offsets = (0, 0)

face_detect = load_detection_model(detect_m_path)

def checkNull():
    if (img_path == None or img_path == '' or currentcmbx_var.get() == '        Select Blur Level'):
        label.config(text= 'ERROR: Image or Blur level not selected')
    elif(str(img_path).lower().endswith(('.png', '.jpg', '.jpeg'))):
        detectFaces()
    else:
        label.config(text= "Selected file is not image ('.png', '.jpg', '.jpeg')")

def detectFaces():
    predicted_image,error,rw, rh = '','','',''
    crdList = []
    crdList2 = []
    rgb_img = load_image(img_path, grayscale=False)
    
    gray_img = load_image(img_path, grayscale=True)
    gray_img = np.squeeze(gray_img)
    gray_img = gray_img.astype("uint8")
    faces = detect_faces(face_detect, gray_img)

    for face_crd_xyz in faces:
           
        x1, x2, y1, y2 = apply_offsets(face_crd_xyz,rgb_offsets)
        rgb_face = rgb_img[y1:y2, x1:x2]
        
        x1, x2, y1, y2 = apply_offsets(face_crd_xyz,gry_offsets)
        grayscale_face = gray_img[y1:y2, x1:x2]

        rgb_face = preprocess_input(rgb_face,False)
        rgb_face = np.expand_dims(rgb_face, 0)

        gray_face = preprocess_input(grayscale_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        crdList.append(x1)
        crdList.append(y1)
        crdList2.append(x2)
        crdList2.append(y2)
        
        rx,ry,rw,rh = face_crd_xyz

    if(len(faces) == 0 or len(faces) == None):
            error = True
    else:
            error = False

    bgr_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('images/blured/_orginal_image.jpg', bgr_image)

    predicted_image = image.load_img('images/blured/_orginal_image.jpg')

    blrFaces(predicted_image,error,crdList,crdList2, rw, rh)

def blrFaces(predicted_image,error,crdList,crdList2, rw, rh):
    clearText = currentcmbx_var.get()
    clearText = clearText.replace("Blur Level:","")
    for i in range(int(len(crdList2)/2)):   
        cropetface = predicted_image.crop((int(crdList[i+i]),int(crdList[i+i+1]),int(crdList2[i+i]),int(crdList2[i+i+1])))
        blur_img = cropetface.filter(ImageFilter.GaussianBlur(radius=int(clearText)))
        predicted_image.paste(blur_img,(int(crdList[i+i]),int(crdList[i+i+1]),int(crdList2[i+i]),int(crdList2[i+i+1])))
        
    imgshow(predicted_image,error)

def imgshow(predicted_image,error):
    figure1 = plt.figure(figsize=(6, 5), dpi=100)
    bar1 = FigureCanvasTkAgg(figure1, mainW)
    bar1.get_tk_widget().place(x=25, y=210)
    ax1 = figure1.add_subplot(111)
    ax1.imshow(predicted_image)
    if(error == True):
        ax1.set_xlabel("Face Cannot Detected")
        ax1.set_xticks([])
        ax1.set_yticks([])
    else:
        ax1.set_xlabel("")
        ax1.set_xticks([])
        ax1.set_yticks([])
    predicted_image.save('images/blured/_blured_image.jpg')
    label.config(text='Image Saved to : images/blured/_blured_image.jpg')
mainloop()