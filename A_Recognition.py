from keras.layers import Input
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import gradient_descent_v2
from model.slowfast import SlowFast_body, bottleneck
import cv2
import numpy as np
import os
from pathlib import Path
import time
from PIL import Image
from PIL import ImageDraw
import telepot
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
ch_id = os.getenv('TELEGRAM_CHAT_ID', '')
bot = telepot.Bot(BOT_TOKEN) if BOT_TOKEN and ch_id else None
def resnet50(inputs, **kwargs):
    model = SlowFast_body(inputs, [3, 4, 6, 3], bottleneck, **kwargs)
    return model
font = cv2.FONT_HERSHEY_SIMPLEX
def frames_from_video(video_dir, nb_frames = 25, img_size = 224):

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
       
        ret, frame = cap.read()
        
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        
        frames.append(frame)
        i+=1
        #cv2.putText(frame, 'TEXT ON VIDEO',(30, 30),font, 1,(0, 255, 255),2,cv2.LINE_4)
        #cv2.imshow('Action Recognition', frame)
        #cv2.waitKey(0)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0

def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])
    out=''
    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))
        out=str(classes[preds[i]])
        if bot is not None:
            bot.sendMessage(ch_id, str(classes[preds[i]]+' detected'))
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        
        frames.append(frame)
        i+=1
        cv2.putText(frame,str(out),(30, 30),font, 1,(0, 255, 255),2,cv2.LINE_4)
        cv2.imshow('Action Recognition', frame)
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()

# Load the model with pre-configured parameters
x = Input(shape = (25, 224, 224, 3))
model = resnet50(x, num_classes=14)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01, momentum=0.9), 
              metrics=['accuracy'])

# model.summary()

# We load the weights directly
ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv('SHAS_MODEL_DIR', str(ROOT))).expanduser().resolve()
model.load_weights(str(MODEL_DIR / "model_new.h5"))
def analyse(fileName):  
    predictions(video_dir = fileName, model = model, nb_frames = 25, img_size = 224)
