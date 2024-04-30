#!/usr/bin/env python
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, session
import os
import datetime
import time
from threading import Condition, Lock


app = Flask(__name__)
app.secret_key = 'DEVELOPMENT_KEY'

dir_notifications = '/home/searchandrescue/notifications/'
listdir_notifications = ['/home/searchandrescue/notifications/life_raft/', '/home/searchandrescue/notifications/life_ring/', '/home/searchandrescue/notifications/life_vest/']

targets = ['life_raft', 'life_ring', 'life_vest']
classes = [0,1,2]

os_lock = Lock()
targets_lock = Lock()

display_message = Condition()
display_timestamp = Condition()
message = ''
timestamp = datetime.time()

#def get_classes():
    #if 'classes' not in g:
        #g.classes = [0,1,2]
    #return g.classes

#@app.teardown_appcontext
#def teardown_classes(exception):
    #classes = g.pop('classes', None)
 
@app.route('/', methods=['GET', 'POST'])
def index():
    selections = ['0','1','2']
    if request.method == 'POST':
        selections = request.form.getlist('target')
    session["classes"] = [eval(i) for i in selections]
    print(selections)                    
    return render_template('index.html', selections=selections)

def stream(cap, model, target_mask):
    while cap.isOpened:
        # Capture frame-by-frame
        success, frame = cap.read()
        
        if success:
            print(target_mask)
            # run model
            results=model.track(frame, classes=target_mask)
        
            # get annotated frame as numpy array
            annotated_frame = results[0].plot()
            
            #cv2.imshow(annotated_frame)
            
            # encode as JPEG and translate to byte stream
            success, encoded_frame = cv2.imencode('.jpeg', annotated_frame)
            if success:
                byte_frame = encoded_frame.tostring()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')
            
            # access shared resource
            with os_lock:
                # save crop
                for result in results:
                    result.save_crop(dir_notifications, datetime.datetime.now().strftime("%H-%M-%S"))
        
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                break
        else:
            cap.release()
            break

def notify():
    while True:
        # go through the folders for each class
        for directory in listdir_notifications:
            # access shared resource
            with os_lock:
                # go through every image saved in the folder
                images = os.listdir(directory)
            if (len(images) != 0):
                for image in images:
                    # get image of object
                    cropped_frame = cv2.imread(directory + image)
                    # encode as JPEG and translate to byte stream
                    success, encoded_frame = cv2.imencode('.jpeg', cropped_frame)
                    if success:
                        byte_frame = encoded_frame.tostring()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')
                    # wait
                    time.sleep(1)
                    # remove image
                    os.remove(directory + image)
            else:
                # delay between os calls
                time.sleep(1)

@app.route('/live_feed')
def live_feed():
    #delay between reloads (allows camera to release)
    time.sleep(5)
    target_mask = session.get("classes", None)
    if target_mask == None:
        target_mask = [0,1,2]
    print(target_mask)
    return Response(stream(cv2.VideoCapture(0), YOLO('best.pt'), target_mask),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/notifications')
def notifications():
    return Response(notify(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.route('/message')
#def message():
    #return Response(send(),
                    #mimetype='text/html')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    
cap.release()