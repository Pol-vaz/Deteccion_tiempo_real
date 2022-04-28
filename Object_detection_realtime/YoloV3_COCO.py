import cv2
import numpy as np
import time


def cargar_modelo_YoloV3(pesos,config,carga_clases):
    model = cv2.dnn.readNet(pesos,config)
    # Da el nombre de las capas del modelo en forma de lista
    layer_names = model.getLayerNames()
    # Devuelve una lista con la posición de las capas que están desconectadas                        
    unconected_layers = model.getUnconnectedOutLayers()            
    output_layers=[]
    # guardo en una lista llamada output_layers los nombres de las capas desconectadas
    for i in unconected_layers:
        n=int(i)
        output_layers.append(layer_names[n-1])
    classes=clases(carga_clases)
    # Devuelve una matriz con valores entre 0 y 255 y de 3 columnas y tantas filas como elementos hay en la lista de CLASSES
    colors = np.random.uniform(0, 255, size=(len(classes), 3))     
    return model,colors,output_layers,classes

def mostrar_fps(imag, start,frames,font):
    #este metodo sirve para algo
    elapsed_time = time.time() - start
    fps = frames / elapsed_time

    cv2.putText(imag, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)


def chorradita():
    print('Me vais a comer el rabillo')

def bounding_boxes(outs,width,height,colors,frame,classes,font,objeto):

    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    for i in range(len(boxes)):
         if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            if objeto is True:
                print('{} detectado al {}% '.format(label,round(confidence*100,2)))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence*100, 2))+"%", (x, y + 30), font, 1, (255,255,255), 1)

def clases(predeterminado=True):
    if predeterminado:
        classes = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
    else:
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
    return classes

def YoloV3(pesos,config,metodo_carga_clases=True,aviso_obj=False): 

    model,colors,output_layers,classes=cargar_modelo_YoloV3(pesos,config,metodo_carga_clases)
    camera = cv2.VideoCapture(-1)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    camera.set(3,1000)
    camera.set(4,800)
    starting_time = time.time()
    frame_id = 0
    objeto=None

    if aviso_obj is True:
        objeto = True

    while True:
        _, frame = camera.read()
        frame_id += 1
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416,416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        outs = model.forward(output_layers)
        bounding_boxes(outs,width,height,colors,frame,classes,font,objeto)
        mostrar_fps(frame,starting_time,frame_id,font)
        cv2.imshow("YoloV3", frame)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            break

    camera.release()
    cv2.destroyAllWindows()

#Cambiar las rutas de los pesos y el archivo de configuracion
pesos= 'yolov3-tiny.weights'
config= 'yolov3-tiny.cfg'
YoloV3(pesos, config, metodo_carga_clases=True,aviso_obj= False)
