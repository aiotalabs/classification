import numpy as np
import argparse
import time
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")

args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
try:
    while True:
        frame = vs.read()
        if frame is None:
            print("[ERROR] Could not read frame from video stream")
            sys.exit(-1)
        frame_rz = imutils.resize(frame,width=32)
        blob = cv2.dnn.blobFromImage(frame_rz, 0.003921, (32,32), (0.4914, 0.4822, 0.4465))
        net.setInput(blob)
        start = time.time()
        preds = net.forward()
        end = time.time()
        #print("[INFO] Classification took {:.5} seconds".format(end-start))
        preds = preds.reshape((1,len(classes)))
        idx = np.argsort(preds[0])[::-1][0]

        text = "Label: {}".format(classes[idx])
        cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()
except KeyboardInterrupt:
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()
    print('[ABORT] Stopping because of KeyboardInterrupt')    

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
