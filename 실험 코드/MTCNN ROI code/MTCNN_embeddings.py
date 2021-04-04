# 사용법
# -i dataset -e embedding/embeddings.pickle -d face_detection_model -m openface_nn4.small2.v1.t7

# 필요한 패키지
from imutils import paths
import argparse
import imutils
import pickle
import cv2
import os
from mtcnn.mtcnn import MTCNN

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

# ROI 추출을 위한 MTCNN 불러오기
print("[INFO] loading face detector...")
detector = MTCNN()

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

	# MTCNN 기반의 ROI를 적용하여 이미지에서 얼굴 위치를 파악
    detections = detector.detect_faces(image)
    
	# 만약 사진에서 얼굴이 한개 이상 검출 될 시
    if len(detections) > 0:
        # JSON 방식으로 저장된 데이터에서 신뢰도 추출
        for detection in detections:
            confidence = detection['confidence']
		    # 추출된 ROI의 신뢰도가 설정한 신뢰도 보다 높을 시
            if confidence > args["confidence"]:
			    # ROI의 x, y 값을 계산
                x, y, w, h = detection['box']
                x1, y1 = x + w, y + h

			    # ROI를 추출하고 ROI의 치수를 설정
                face = image[y:y1, x:x1]
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