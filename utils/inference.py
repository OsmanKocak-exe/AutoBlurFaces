import cv2
from keras.preprocessing import image

'''
def load_image(image_path, grayscale=False,color_mode='rgb', target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size, color_mode)
    return image.img_to_array(pil_image)
'''
def load_image(image_path, grayscale=False, target_size=None):
    color_mode = 'grayscale'
    if grayscale == False:
        color_mode = 'rgb'
    else:
        grayscale = False
    pil_image = image.load_img(image_path, grayscale, color_mode, target_size)
    
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)
    
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)
