import numpy as np
import cv2 as cv
import time

def Mobilenet_SSD_OpenCV(pesos,config):

    classes = ["background", "person", "bicycle", "car", "motorcycle",
                    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                    "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                    "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                    "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                    "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

    colors= np.random.uniform(0,255, size=(len(classes),3))

    camera = cv.VideoCapture(-1)
    camera.set(3,1000)
    camera.set(4,800)
    font = cv.FONT_HERSHEY_COMPLEX_SMALL
    cvNet = cv.dnn.readNetFromTensorflow(pesos,config)
    starting_time = time.time()
    frame_id = 0

    while True:


        _, img = camera.read()
        frame_id+=1
        rows = img.shape[0]
        cols = img.shape[1]
        cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        cvOut = cvNet.forward()

        for detection in cvOut[0,0,:,:]:
            
            scores = float(detection[2])
            classId = int(detection[1])
            label = classes[classId]
            
            if scores > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                color =colors[classId]
                cv.rectangle(img,(int(left), int(top)), (int(right), int(bottom)), color , thickness=2)

                cv.rectangle(img, (int(left), int(top)), (int(right),( int(top) + 25)), color, -1)
                cv.putText(img, str(label) + " " + str(round(scores*100, 2))+"%" , (int(left), int(top + 25)),1,1, (255,255,255), 1)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)

        cv.imshow('MobileNet SSD OpenCV', img)
        if cv.waitKey(1) & 0xFF == ord("s"):
                break

    camera.release()
    cv.destroyAllWindows()
