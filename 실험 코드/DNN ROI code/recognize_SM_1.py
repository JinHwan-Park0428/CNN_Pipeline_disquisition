# USAGE
# S
import numpy as np
import argparse
import imutils
import pickle
import cv2
import tensorflow as tf
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-k", "--keras", required=True,
    help="path to load DNN train data")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
le = pickle.loads(open(args["le"], "rb").read())
model = tf.keras.models.load_model(args["keras"])

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# loop over the detections

	# extract the confidence (i.e., probability) associated with the
	# prediction

    
	# filter out weak detections

		# compute the (x, y)-coordinates of the bounding box for the
		# face
box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype("int")
        #print(box)

		# extract the face ROI
face = image[startY:endY, startX:endX]
(fH, fW) = face.shape[:2]
        
        
        

		# ensure the face width and height are sufficiently large
#if fW < 20 or fH < 20:
  #continue


		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
embedder.setInput(faceBlob)
vec = embedder.forward()

		# perform classification to recognize the face
preds = model.predict(np.array(vec))[0]

j = np.argmax(preds)
proba = preds[j] 
name = le.classes_[j]
        
        
       
		# draw the bounding box of the face along with the associated
		# probability
text = "{}: {:.2f}%".format(name, proba * 100)
y = startY - 10 if startY - 10 > 10 else startY + 10
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
"""
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate('C:/Users/USER/Desktop/face-recognition-pj-firebase-adminsdk-h9tow-cf5b44e3b2.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://face-recognition-pj.firebaseio.com/',
    'storageBucket': 'face-recognition-pj.appspot.com'
})
    
from uuid import uuid4
if(name == "unkowns"):
    str1 = "unknowns"
else:
    str1 = "knowns"
    
bucket = storage.bucket()
str2 = str1 + ".jpg"
blob = bucket.blob(str2)
# Create new token
new_token = uuid4()
# Create new dictionary with the metadata
metadata  = {"firebaseStorageDownloadTokens": new_token}
# Set metadata to blob
blob.metadata = metadata
# Upload file
cv2.imwrite('images/temp.jpg', image)
blob.upload_from_file("C:/Users/USER/opencv-face-recognition/images/temp.jpg") #<-error (AttributeError: 'str' object has no attribute 'tell')
import numpy as np
import  time
ref = db.reference(str1)
ref.update({
    'names': name,
    'time': time.strftime('%c', time.localtime(time.time())),
    'image': "https://firebasestorage.googleapis.com/v0/b/for-python-952b1.appspot.com/o/"+str1+".jpg?alt=media"
})
"""