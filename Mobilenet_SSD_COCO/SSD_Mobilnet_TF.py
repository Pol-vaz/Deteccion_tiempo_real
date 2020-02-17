
import numpy as np
import tensorflow as tf
import cv2 as cv
import time

def Mobilenet_SSD_Tensorflow(pesos):

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

    # Read the graph.
    with tf.io.gfile.GFile(pesos, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        cap= cv.VideoCapture(-1)
        font = cv.FONT_HERSHEY_COMPLEX_SMALL
        cap.set(3,1000)
        cap.set(4,800)

        starting_time = time.time()
        frame_id = 0
        while True:
            # Read and preprocess an image.
            _,img = cap.read()
            frame_id += 1
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                label = classes[classId]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    color =colors[classId]
                    cv.rectangle(img,(int(x), int(y)), (int(right), int(bottom)), color , thickness=2)

                    cv.rectangle(img, (int(x), int(y)), (int(right),( int(y) + 30)), color, -1)
                    cv.putText(img, str(label) + " " + str(round(score*100, 2))+"%" , (int(x), int(y + 25)),1,1, (255,255,255), 1)
                    

                    elapsed_time = time.time() - starting_time
                    fps = frame_id / elapsed_time
            cv.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)
            cv.imshow('MobileNet SSD Tensorflow', img)

            if cv.waitKey(1) & 0xFF == ord("s"):
                break

    cap.release()
    cv.destroyAllWindows()