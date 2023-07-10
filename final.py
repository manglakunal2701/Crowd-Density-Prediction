import numpy as np
import cv2
import os
from tkinter import *
import matplotlib.pyplot as plt
from scipy.spatial import distance as distance
import cmath
from tkinter import filedialog
from tkinter import messagebox

 
root = Tk()
root.title(" People Density Estimation ")
root.configure(background="lightgreen")
root.geometry('1200x1150')
labelpath = r'C:\Users\Kunal Mangla\Desktop\python\crowd computing\coco.names'
file = open(labelpath)
label = file.read().strip().split("\n")

file.close()


weightspath = r'C:\Users\Kunal Mangla\Desktop\python\crowd computing\yolov3.weights'
configpath = r'C:\Users\Kunal Mangla\Desktop\python\crowd computing\yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
layer_names = net.getLayerNames()
ln = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


def videocheck():
    i=0

    fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Open file",filetypes=(("MP4","*.mp4"),("All File","*.*")))
    if(len(fln)==0):
      messagebox.showerror('Error','The video is not selected please select the video',parent=root)
      return
    videopath =fln
    video = cv2.VideoCapture(videopath)
    data=[]
    while(True):
        ret, frame = video.read()
        if ret == False:
            print('Error running the file :(')
        frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        net.setInput(blob)
        
        outputs = net.forward(ln)
        

        boxes = []
        confidences = []
        classIDs = []
        center = []
        output = []
        count = 0
        results = []
        

        h, w = frame.shape[:2]
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
            
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    center.append((centerX, centerY))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  
       
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
               
                if(label[classIDs[i]] == 'person'):
                    
                    cX = (int)(x+(y/2))
                    cY = (int)(w+(h/2))
                    center.append((cX, cY))
                    res = ((x, y, x+w, y+h), center[i])
                    results.append(res)
                    dist = cmath.sqrt(
                        ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                    if(dist.real < 100):
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 0, 255), 2)
                        cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                        count = count+1

                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)
                        cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                        count = count+1
            
           

            cv2.putText(frame, "Count: {}".format(
                count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
        cv2.imshow('Frame', frame)
        
        print(count)
        current_time =i
        data.append((count,current_time))
        i=i+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print(1)
    
    video.release()
    cv2.destroyAllWindows()
    if count >0:
        t3.delete("1.0", END)
        t3.insert(END, count)
    
    def Sort(sub_li):
        sub_li.sort(key = lambda x: x[1])
        return sub_li
    print(Sort(data))
    print(data)
    x = []
    y=[]
    for i in data:
        x.append(i[1])
    for i in data:
        y.append(i[0])
    

    # graph plot
    
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Count of people')
    plt.title('Count vs time!')  
    plt.show()

def photo():

    ret = True
    f_types= [('Image Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    if(len(filename)==0):
      messagebox.showerror('Error','The photo is not selected please select the photo',parent=root)
      return 
    img=cv2.imread(filename)
    frame=img
    cv2.imshow('Frame', frame)
    if ret == False:
        print('Error running the file :(')
    frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    net.setInput(blob)
    
    outputs = net.forward(ln)
    

    boxes = []
    confidences = []
    classIDs = []
    center = []
    output = []
    count = 0
    results = []

    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)

            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                center.append((centerX, centerY))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
           
            if(label[classIDs[i]] == 'person'):
                cX = (int)(x+(y/2))
                cY = (int)(w+(h/2))
                center.append((cX, cY))
                res = ((x, y, x+w, y+h), center[i])
                results.append(res)
                dist = cmath.sqrt(
                    ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                if(dist.real < 100):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                    count = count+1

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                    count = count+1
        
        cv2.putText(frame, "Count: {}".format(
            count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    
    cv2.imshow('Frame', frame)
   
    if count >0:
        t4.delete("1.0", END)
        t4.insert(END, count)
    cv2.waitKey()
    cv2.destroyAllWindows()



def Sort(sub_li):
  
   
    sub_li.sort(key = lambda x: x[1])
    return sub_li


w2 = Label(root,justify=LEFT, text=" People density Estimation using Machine Learning ")
w2.config(font=("Elephant", 30),background="lightblue")
w2.grid(row=1, column=0, columnspan=2, padx=100,pady=40)
lr = Button(root, text="Video",height=2, width=10, command=videocheck)
lr.config(font=("Elephant", 12),background="green")
lr.grid(row=15, column=0,pady=20)
lr = Button(root, text="Photo",height=2, width=10, command=photo)
lr.config(font=("Elephant", 12),background="green")
lr.grid(row=16, column=0,pady=20)

NameLb = Label(root, text="Predict using photo or video :")
NameLb.config(font=("Elephant", 15),background="lightblue")
NameLb.grid(row=13, column=0, pady=20)

NameLb = Label(root, text="Output :")
NameLb.config(font=("Elephant", 15),background="lightblue")
NameLb.grid(row=13, column=1, pady=20)

t3 = Text(root, height=2, width=15)
t3.config(font=("Elephant", 15))
t3.grid(row=15, column=1 ,padx=60)
t4 = Text(root, height=2, width=15)
t4.config(font=("Elephant", 15))
t4.grid(row=16, column=1 ,padx=60)

root.mainloop()
