dataset = 훈련 데이터가 들어가는 폴더 입니다. 폴더 내부에 라벨로 사용할 이름이 적인 폴더를 만들고 사진을 집어 넣으시면 됍니다. EX) ./dataset/JinHwan/1.jpg

embedding = 임베딩 코드에서 추출한 임베딩을 저장하는 폴더입니다.

face_detection_model = DNN 방식의 ROI 추출을 위한 caffe 모델이 들어있는 파일입니다.

openface_nn4.small2.v1.t7 = 추출된 ROI에서 임베딩을 하기 위한 학습된 모델입니다.

images = 검증을 위한 데이터가 들어가는 폴더 입니다. 내부 방식은 dataset 폴더와 같습니다. EX) ./images/JinHwan/1.jpg

SVM = SVM 알고리즘을 이용해 훈련 시킬 경우 Label과 훈련 결과물이 저장되는 폴더입니다.

SM = SM 알고리즘을 이용해 훈련 시킬 경우 Label과 훈련 결과물이 저장되는 폴더입니다.

DNN ROI code = DNN방식의 ROI를 이용한 Face Recognition 파이썬 코드가 들어있습니다. 

MTCNN ROI code = MTCNN방식의 ROI를 이용한 Face Recognition 파이썬 코드가 들어있습니다.

