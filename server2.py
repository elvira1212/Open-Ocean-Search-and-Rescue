#!/usr/bin/env python
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, request
import os
import datetime
import time
from threading import Condition, Lock

app = Flask(__name__)

dir_notifications = '/home/searchandrescue/notifications/'
listdir_notifications = ['/home/searchandrescue/notifications/life_raft/', '/home/searchandrescue/notifications/life_ring/', '/home/searchandrescue/notifications/life_vest/']

mutex = Lock()
display_message = Condition()
display_timestamp = Condition()
message = ''
timestamp = datetime.time()
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select')
def select():
    if request.method == 'POST':
        print(request.form.getlist('target'))

def stream(cap, model):   
    while cap.isOpened:
        # Capture frame-by-frame
        success, frame = cap.read()
        # Get target objects
        #targets = request.form.getlist('target')
        
        if success:
            # run model
            results=model.track(frame)
        
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
            with mutex:
                # save crop
                for result in results:
                    result.save_crop(dir_notifications, 'img')
        
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

def notify():
    while True:
        # go through the folders for each class
        for directory in listdir_notifications:
            # access shared resource
            with mutex:
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
                
#def send():
    #while True:
        #yield (b'')
        #display_message.wait()

@app.route('/live_feed')
def live_feed():
    return Response(stream(cv2.VideoCapture(0), YOLO('best.pt')),
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