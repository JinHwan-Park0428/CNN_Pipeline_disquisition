# 사용법
# -i dataset -e embedding/embeddings8_3folders.pickle -d face_detection_model -m openface_nn4.small2.v1.t7

# 필요한 패키지
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# 필요한 파일을 일종의 명령문을 이용하여 설정
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
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

# 훈련 데이터 불러오기
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# 임베딩과 라벨을 집어넣을 변수 초기화
knownEmbeddings = []
knownNames = []

# 얼굴 검출 횟수를 체크하기 위한 변수 초기화
total = 0

# 훈련 데이터 갯수 만큼 반복
for (i, imagePath) in enumerate(imagePaths):
	# 훈련 데이터에서 라벨 추출
    print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

	# 이미지를 불러오고, 가로폭이 600픽셀이 되도록
    # 크기를 조정(세로비는 유지하면서)한 다음 이미지의 치수를 설정
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

	# 이미지 블롭데이터를 저장
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# DNN 기반의 ROI를 적용하여 이미지에서 얼굴 위치를 파악
    detector.setInput(imageBlob)
    detections = detector.forward()

	# 만약 사진에서 얼굴이 한개 이상 검출 될 시
    if len(detections) > 0:
		# 훈련 데이터에는 얼굴이 한개만 있을 것이기 때문에, 신뢰도가 가장 높은 ROI만을 추출
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

		# 추출된 ROI의 신뢰도가 설정한 신뢰도 보다 높을 시
        if confidence > args["confidence"]:
			# ROI의 x, y 값을 계산
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# ROI를 추출하고 ROI의 치수를 설정
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

			# 단 추출한 ROI가 설정한 크기보다 커야 진행
            if fW < 20 or fH < 20:
                continue

			# 추출한 ROI의 이미지 블롭 데이터를 저장하고
            # 불러왔던 임베딩 모델을 통해 128개의 특징을 추출
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
			    (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

			# 추출된 라벨과 임베딩 데이터를 저장
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# 임베딩 데이터와 라벨을 pickle 파일을 통해 저장
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()