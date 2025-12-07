import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# ============================
# 🔧 PATH DATASET
# ============================
TRAIN_PATH = "dataset/images"
MASK_PATH = "dataset/masks"

train_files = sorted(os.listdir(TRAIN_PATH))
mask_files = sorted(os.listdir(MASK_PATH))

print(f"Found {len(train_files)} images and {len(mask_files)} masks")

# ============================
# 🔧 Preprocessing
# ============================
IMG_SIZE = 128
X, Y = [], []

for img_name, mask_name in zip(train_files, mask_files):
    img = cv2.imread(f"{TRAIN_PATH}/{img_name}")
    mask = cv2.imread(f"{MASK_PATH}/{mask_name}", cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = np.expand_dims(mask / 255.0, axis=-1)

    X.append(img)
    Y.append(mask)

X = np.array(X, dtype="float32")
Y = np.array(Y, dtype="float32")

print("Dataset Loaded:", X.shape, Y.shape)

# ========================
# 🔧 Train-test split
# ========================
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

# ========================
# 🔧 Loss & Metrics
# ========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ========================
# 🔧 UNET Lite Model
# ========================
def unet_lite(input_size=(128,128,3)):
    inputs = Input(input_size)

    c1 = Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = MaxPooling2D(2)(c3)

    b = Conv2D(128, 3, activation="relu", padding="same")(p3)
    b = Conv2D(128, 3, activation="relu", padding="same")(b)

    u3 = Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    u3 = concatenate([u3, c3])
    c6 = Conv2D(64, 3, activation="relu", padding="same")(u3)

    u2 = Conv2DTranspose(32, 2, strides=2, padding="same")(c6)
    u2 = concatenate([u2, c2])
    c7 = Conv2D(32, 3, activation="relu", padding="same")(u2)

    u1 = Conv2DTranspose(16, 2, strides=2, padding="same")(c7)
    u1 = concatenate([u1, c1])
    c8 = Conv2D(16, 3, activation="relu", padding="same")(u1)

    outputs = Conv2D(1, 1, activation="sigmoid")(c8)

    return Model(inputs, outputs)

model = unet_lite()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=bce_dice_loss,
              metrics=["binary_accuracy", dice_coef])

model.summary()

# ========================
# 🔧 Train Model
# ========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=4,
    callbacks=callbacks
)

# ========================
# 💾 SAVE MODEL
# ========================
model.save("model_unet.h5")
print("\n✅ Model saved as model_unet.h5")
