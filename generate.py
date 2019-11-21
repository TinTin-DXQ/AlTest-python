import copy
import random
import numpy as np
from tensorflow import keras


#读取模型
model = keras.models.load_model("MyModel.h5")

def average(img):
    res=0.0
    sum = 0.0
    for i in range(28):
        for j in range(28):
            sum=sum+ img[i][j]
    res=sum/(28*28)
    return res

def deviation(img, u):
    res=0.0
    d = 0.0
    for i in range(28):
        for j in range(28):
            d = d + (img[i][j] - u) * (img[i][j] - u)
    res=(d/(28*28 - 1))**0.5
    return res

def assit_function(img1, u1, img2, u2):
    res=0.0
    d = 0.0
    for i in range(28):
        for j in range(28):
            d += (img1[i][j] - u1) * (img2[i][j] - u2)
    res=d/(28*28 - 1)
    return res

def SSIM(img1 ,img2):
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1*L)*(K1*L)
    C2 = (K2*L)*(K2*L)
    C3 = C2/2

    u1 = average(img1)
    u2 = average(img2)
    d1=deviation(img1, u1)
    d2 = deviation(img2, u2)
    d12 = assit_function(img1, u1, img2, u2)

    L = (2*u1*u2 + C1)/(u1*u1 + u2*u2 + C1)
    C = (2 * d1 * d2 + C2) / (d1 * d1 + d2 * d2 + C2)
    S = (d12 + C3)/(d1*d2 + C3)
    return L*C*S


def one_attack(img):
    count=0
    while count<=784:
        rImg=copy.copy(img)
        for n in range(5):
            x=random.randint(0,27)
            y=random.randint(0,27)
            rImg[x][y]=random.random()
        before=np.argmax(model.predict(np.expand_dims(img,0)))
        after=np.argmax(model.predict(np.expand_dims(rImg,0)))
        if before==after:
            count=count + 1
        else:
            print("one!!!")
            return rImg
    return img

def two_attack(img):
    count = 0
    while count <= 30:
        rImg = copy.copy(img)
        for n in range(10):
            x1 = random.randint(0, 27)
            y1 = random.randint(0, 27)
            x2=random.randint(0,27)
            y2=random.randint(0,27)
            temp=rImg[x1][y1]
            rImg[x1][y1] = rImg[x2][y2]
            rImg[x2][y2]=temp
        before = np.argmax(model.predict(np.expand_dims(img, 0)))
        after = np.argmax(model.predict(np.expand_dims(rImg, 0)))
        if before == after:
            count = count + 1
        else:
            print("two!!!")
            return rImg
    return img

def last_attack(img):
    rImg=copy.copy(img)
    for i in range(25):
        for j in range(25):
            rImg[i][j]=random.random()
    print("three")
    return rImg

def generate(images,shape):
    print("fdfsdf")

    generate_images=np.empty_like(images)
    test_images=images/255.0
    count=0
    for test_image in test_images:
        print(count)
        if(count>10):
            break
        cImg=one_attack(test_image)
        before=np.argmax(model.predict(np.expand_dims(test_image, 0)))
        after=np.argmax(model.predict(np.expand_dims(cImg, 0)))
        if(before==after):
            cImg=two_attack(test_image)
            before = np.argmax(model.predict(np.expand_dims(test_image, 0)))
            after = np.argmax(model.predict(np.expand_dims(cImg, 0)))
            if(before==after):
                cImg = last_attack(test_image)
        generate_images[count] = cImg
        count=count+1
    return generate_images

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
generate(test_images,(10000,28,28,1))