# 사용법
# -e embedding/embeddings.pickle -l SVM/le.pickle -r SVM/recognizer.pickle 

# 필요한 패키지
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# 필요한 파일을 일종의 명령문을 이용하여 설정
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# 추출된 임베딩 피클 불러오기
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# 추촐된 라벨 불러오기
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# 추출된 128개의 임베딩을 이용하여 검증 데이터에 사용될 모델을 SVM 알고리즘을 이용하여 훈련
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# 훈련된 모델을 피클 파일로 저장
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# 불러온 라벨을 피클 파일로 저장
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()