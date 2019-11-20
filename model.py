from tensorflow import keras
import ssl

def create_model():
    # 构建模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=5)
    return model

#加载数据返回4个28*28的NumPy数组
ssl._create_default_https_context=ssl._create_unverified_context
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_images.shape)
#print(len(train_labels))
#print(train_labels)

#每个像素点的取值范围为[0,1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# 评估准确率
model = create_model()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)

model.save('MyModel.h5')