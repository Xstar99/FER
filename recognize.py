import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from getdata import preprocess_input
from analysis import *

class Emotion_rec:
    def __init__(self):
        # 检测xml路径、表情识别路径
        detection_model_path = 'detection/haarcascade_frontalface_alt.xml'
        emotion_model_path = 'Model/mini_XCEPTION.hdf5'

        # 人脸检测模型(级联分类器)
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        # 人脸表情识别模型
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # 表情标签
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    
    def run(self, frame_in):
        """
        frame_in 摄像画面或图像
        label_face 用于人脸显示画面的label对象
        label_result 用于显示结果的label对象
        """

        # 调节画面大小
        frame = cv2.resize(frame_in, (300,300))  # 缩放画面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 检测人脸
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        preds = []
        label = None
        (fx, fy, fw, fh) = None, None, None, None
        # 复制画面
        frameClone = frame.copy()

        if len(faces) > 0:
            # ROI
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序

            for i in range(len(faces)):  # 遍历每张检测到的人脸，默认识别全部人脸
                
                (fX, fY, fW, fH) = faces[i]
                # 从灰度图中提取感兴趣区域（ROI），将其大小转换为与模型输入相同的尺寸，并为通过CNN的分类器准备ROI
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # 用模型预测各分类的概率
                preds = self.emotion_classifier.predict(roi)[0]
                # emotion_probability = np.max(preds)  # 最大的概率
                label = self.EMOTIONS[preds.argmax()]  # 选取最大概率的表情类

                # 圈出人脸区域并显示识别结果
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)
        # 表情分析
        emotion_analysis(self.EMOTIONS, preds)