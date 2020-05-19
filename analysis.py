import numpy as np
import matplotlib.pyplot as plt

def emotion_analysis(emotion, preds):
    y_pos = np.arange(len(emotion))

    plt.bar(y_pos, preds, align='center', alpha=0.5)
    plt.xticks(y_pos, emotion)
    plt.ylabel('percentage')
    plt.title('emotion')
 
    plt.show()