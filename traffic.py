import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

EPOCHS = 50
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    dataset_path = "gtsrb//0//"    #路径需自定义

    # Get image arrays and labels for all image files
    images, labels = load_data(dataset_path)
    # Split data into training and testing sets
    # labels = tf.keras.utils.to_categorical(labels)            #将类别标签转换为onehot编码
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

#数据载入
#-------------------------------
def load_data(data_dir):
    X=[]
    Y=[]
    for detail in range (0,42):

        data_dir=os.path.join('gtsrb/'+str(detail)+'/')         #os.path.join()的用法
        for i in glob.glob(data_dir+'*.ppm',recursive=True):    #遍历文件夹，读出数据。

            image=cv2.imread(i)
            img_resize = cv2.resize(image, (36, 36))          #全都变成同一个大小
            X.append(img_resize)
            Y.append(int(detail))
                                            #tuple元组和列表list相似，但不可变
    return (X,Y)
    raise NotImplementedError


#网络模型
#------------------------------------------
def get_model():

    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32,(3,3),activation="relu",input_shape=(36,36,3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(43,activation="softmax")
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model








    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
