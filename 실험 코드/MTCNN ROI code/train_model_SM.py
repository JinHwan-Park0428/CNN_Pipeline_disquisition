# 사용법
# -e embedding/embeddings.pickle -l SM/le.pickle -m SM/model.h5

# 필요한 패키지
from sklearn.preprocessing import LabelEncoder
import argparse
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트

# 필요한 파일을 일종의 명령문을 이용하여 설정
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
ap.add_argument("-m", "--model", required=True,
    help="path to output DNN train data")
args = vars(ap.parse_args())

# 추출된 임베딩 피클 불러오기
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# 입력 값
dataX = data["embeddings"]
dataX = np.array(dataX)

# 추촐된 라벨 불러오기
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# one-hot
dataY = to_categorical(labels)

# 추출된 128개의 임베딩을 이용하여 검증 데이터에 사용될 모델을 SM 알고리즘을 이용하여 훈련
print("[INFO] training model...")
model = Sequential()

#DNN1(input node-128, output node- 라벨 갯수) 
#model.add(Dense(8, input_dim = 128, activation = 'softmax'))

#DNN2(input node-128, hidden node-128, output node- 라벨 갯수)  
#model.add(Dense(128, input_dim = 128,  kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(8, kernel_initializer='uniform', activation = 'softmax'))

sgd = optimizers.SGD(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#훈련된 모델을 h5파일로 저
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(args["model"], save_best_only=True)

history = model.fit(dataX, dataY, validation_data=(dataX, dataY), batch_size=1, epochs=100, callbacks=[modelCheckpoint])

# 불러온 라벨을 피클 파일로 저장
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
