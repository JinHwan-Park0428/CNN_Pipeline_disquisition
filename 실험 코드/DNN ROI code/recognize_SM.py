# 사용법
# -d face_detection_model -m openface_nn4.small2.v1.t7 -k SM/model.h5 -l SM/le.pickle -i images/JinHwan/1.jpg

# 필요한 패키지
import numpy as np
import argparse
import imutils
import pickle
import cv2
import tensorflow as tf
import os

# 필요한 파일을 일종의 명령문을 이용하여 설정
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
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 추출을 위한 Caffe 모델 불러오기
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 임베딩을 위한 임베딩 모델 불러오기
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 훈련된 모델과 라벨을 불러오기
le = pickle.loads(open(args["le"], "rb").read())
model = tf.keras.models.load_model(args["keras"])

# 이미지를 불러오고, 가로폭이 600픽셀이 되도록
# 크기를 조정(세로비는 유지하면서)한 다음 이미지의 치수를 설정
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# 이미지 블롭데이터를 저장
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# DNN 기반의 ROI를 적용하여 이미지에서 얼굴 위치를 파악
detector.setInput(imageBlob)
detections = detector.forward()

# 발견된 ROI 만큼 반복
for i in range(0, detections.shape[2]):
	# ROI에서 신뢰도 추출
    confidence = detections[0, 0, i, 2]
    
	# 추출된 ROI의 신뢰도가 설정한 신뢰도 보다 높을 시
    if confidence > args["confidence"]:
		# ROI의 x, y 값을 계산
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
       
		# ROI를 추출하고 ROI의 치수를 설정
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        
		# 단 추출한 ROI가 설정한 크기보다 커야 진행
        if fW < 20 or fH < 20:
          continue

		# 추출한 ROI의 이미지 블롭 데이터를 저장하고
        # 불러왔던 임베딩 모델을 통해 128개의 특징을 추출
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

		# 훈련 모델을 이용하여 분류 실시
        preds = model.predict(np.array(vec))[0]
        j = np.argmax(preds)
        proba = preds[j] 
        name = le.classes_[j]
        
		# 추출된 확률과 ROI를 사진에 표시
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 이미지 출력
cv2.imshow("Image", image)
cv2.waitKey(0)