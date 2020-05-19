from recognize import Emotion_rec
import numpy as np
import cv2

if __name__ == "__main__":
    filePath = 'testimage/John.jpg'

    image = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    emotion_rec = Emotion_rec()
    emotion_rec.run(image)