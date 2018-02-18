from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import cv2
import imutils
import numpy as np
import playsound
import tensorflow as tf
from imutils.video import VideoStream
from tensorflow.python.platform import gfile


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def load(path):
    image = cv2.imread(path)
    print(image.shape)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    faceCascade = cv2.CascadeClassifier(
        'C:/Users/sanka/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('C:/Users/sanka/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml')
    return (image, gray, faceCascade, eyeCascade)


def display(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image(image, input_height, input_width, input_mean, input_std, sess):
    float_caster = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = sess.run(normalized)
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def classify(eyes):
    model_file = "C:/Users/sanka/Desktop/drowsiness-detection/drowsiness-detection/retrain/eyes/retrained_graph.pb"
    label_file = "C:/Users/sanka/Desktop/drowsiness-detection/drowsiness-detection/retrain/eyes/retrained_graph.txt"
    input_height = 299#224
    input_width = 299
    input_mean = 128
    input_std = 128
    input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    print("loaded model..........")

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    # labels = load_labels(label_file)
    labels = ['closedeyes','openeyes']
    pred = []
    with tf.Session(graph=graph) as sess:
        print("session started.....")
        for eye in eyes:
            t = read_tensor_from_image(eye, input_height=input_height, input_width=input_width, input_mean=input_mean,
                                       input_std=input_std, sess=sess)
            try:
                results = sess.run(output_operation.outputs[0],feed_dict={input_operation.outputs[0]: t})
            except Exception:
                print(Exception.with_traceback())
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]
            pred.append(labels[top_k[0]])
    return pred


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--alarm", type=str, default="",
                    help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())

    COUNTER = 0
    ALARM_ON = False
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/sanka/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/sanka/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml')

    # loop over frames from the video stream
    ans=[]
    try:
        video = []
        video_eye = []
        coord = []
        start = time.time()
        while True:
            frame = vs.read()
            frame = imutils.resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if (len(eyes) > 0):

                    for i in range(0, len(eyes)):
                        (ex, ey, ew, eh) = eyes[i]
                        coord.append((ex + x, ey + y, ew, eh))
                    (ex, ey, ew, eh) = eyes[0]
                    temp_eyes = frame[ey + y:(ey + eh + y), ex + x:(x + ex + ew)]
                    # display(temp_eyes)
                    cv2.rectangle(frame, (ex + x, ey + y), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    video_eye.append(temp_eyes)
                    ans.append(0)
                else:
                    ans.append(1)
                video.append(frame)

            end = time.time() - start
            if (end > 10):
                break
        print("video recorded")
        print("frames are {0}".format(len(video)))

        cv2.destroyAllWindows()
        vs.stop()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Be sure to use lower case
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
        print(video[0].shape)


        prediction = classify(video_eye)

        print("done prediction")
        j=0
        for i in range(len(ans)):
            if(ans[i]==0):
                ans[i]=prediction[j]
                j+=1
            else:
                ans[i]="closedeyes"

        for i in range(len(video)):
            frame=video[i]
            text = ans[i]
            cv2.putText(frame, text, (50, 50), cv2.FONT_ITALIC, 0.8, 255)
            out.write(frame)

        out.release()

    except Exception:
        cv2.destroyAllWindows()
        vs.stop()

    finally:
        cv2.destroyAllWindows()
        vs.stop()
