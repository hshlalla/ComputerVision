#----------------------------------------------------------** setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(device_lib.list_local_devices())
#----------------------------------------------------------** Load the data: the cats vs dogs dataset
#!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
#!unzip -q kagglecatsanddogs_5340.zip
# !ls

#----------------------------------------------------------** Filter out corrupted images (손상된 이미지 전처리)
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path,fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            #큐를 옮겨 10바이트 를가져온다.
            #in을 사용했으므로
            #헤더에 jfif가 없으면 손상된 파일이므로 이미지를 구분하기 전에 전처리 한다.
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped +=1
            # Delete corrupted image
            os.remove(fpath)
print("Deleted %d images" % num_skipped)


#----------------------------------------------------------** 데이터 셋 생성
image_size = (180,180)
batch_size = 8

train_ds = tf.keras.preprocessing.image_dataset_from_directory("PetImages",
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=1337,
                                                               image_size=image_size,
                                                               batch_size=batch_size,
                                                               )
val_ds = tf.keras.preprocessing.image_dataset_from_directory("PetImages",
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=1337,
                                                             image_size=image_size,
                                                             batch_size=batch_size,
                                                             )

#----------------------------------------------------------** 이미지 확인

# plt.figure(figsize=(10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax= plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")
#         plt.show()

#----------------------------------------------------------** 이미지 데이터 증강
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
# plt.figure(figsize=(10,10))
# for images,_ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3,3,i+1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")
#         plt.show()
#

#----------------------------------------------------------**  데이터 표준화 standardization
#option 1
# inputs = keras.Input(shape=input_shape)
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)
#option 2
# augmented_train_ds = train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True),y))

##----------------------------------------------------------** Configure the dataset for performance
train_ds=train_ds.prefetch(buffer_size=32)
val_ds=val_ds.prefetch(buffer_size=32)

##----------------------------------------------------------** 모델구축
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0/255)(x)
    x = layers.Conv2D(32,3,strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64,3,padding = "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128,256,512,728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size,1,strides = 2, padding ="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(1024,3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = 1

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size+(3,),num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,epochs=epochs,callbacks=callbacks,validation_data=val_ds
)

img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg",target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array,0)

predictions = model.predict(img_array)
score = predictions[0]
print(
    "this image is %.2f percent cat and %.2f percent dog."
    %(100*(1-score),100*score)
)