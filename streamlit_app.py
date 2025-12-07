import streamlit as st

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

# Dataset Folder
MASK_PATH = 'Masks'
TRAIN_PATH = 'Images'

mask_files = os.listdir(MASK_PATH)
train_files = os.listdir(TRAIN_PATH)

# ==== ✅ Preview Dataset (JANGAN DIUBAH) ====
for i in range(8, 12):
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(f"{TRAIN_PATH}/{train_files[i]}"))
    plt.title('Gambar')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.imread(f"{MASK_PATH}/{mask_files[i]}"))
    plt.title('Label/Mask')
    plt.show()


# ============ ✅ Preprocessing Dataset ============
IMG_SIZE = 128

X = []
Y = []

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



# ✅ Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

print(X_train.shape, X_val.shape)

# =================✅ Loss Functions & Metrics ==================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# =================✅ UNet (Lite Version – CPU Friendly) =============
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def unet_lite(input_size=(128,128,3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)

    b = Conv2D(128, 3, activation='relu', padding='same')(p3)
    b = Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = Conv2DTranspose(64, 2, strides=2, padding='same')(b)
    u3 = concatenate([u3, c3])
    c6 = Conv2D(64, 3, activation='relu', padding='same')(u3)

    u2 = Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    u2 = concatenate([u2, c2])
    c7 = Conv2D(32, 3, activation='relu', padding='same')(u2)

    u1 = Conv2DTranspose(16, 2, strides=2, padding='same')(c7)
    u1 = concatenate([u1, c1])
    c8 = Conv2D(16, 3, activation='relu', padding='same')(u1)

    outputs = Conv2D(1, 1, activation='sigmoid')(c8)

    return Model(inputs, outputs)


model = unet_lite()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=bce_dice_loss,
              metrics=["binary_accuracy", iou, dice_coef])

model.summary()

# =================✅ Training ==================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=4,
    callbacks=callbacks
)


import matplotlib.pyplot as plt

# ==========================
# 📈 Plot Loss
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 📈 Plot IoU
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history["iou"], label="Train IoU")
plt.plot(history.history["val_iou"], label="Val IoU")
plt.title("Training vs Validation IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 📈 Plot Dice Coef
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history["dice_coef"], label="Train Dice Coef")
plt.plot(history.history["val_dice_coef"], label="Val Dice Coef")
plt.title("Training vs Validation Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Dice Coef")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 📈 Plot Binary Accuracy
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history["binary_accuracy"], label="Train Binary Acc")
plt.plot(history.history["val_binary_accuracy"], label="Val Binary Acc")
plt.title("Training vs Validation Binary Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# # =================✅ Evaluation ==================
# scores = model.evaluate(X_val, Y_val, verbose=1)
# print(f"\n✅ Evaluation Results:\n"
#       f"Binary Accuracy: {scores[1]:.4f}\n"
#       f"IoU: {scores[2]:.4f}\n"
#       f"Dice Coef: {scores[3]:.4f}")


# # =======================
# # 🔍 Evaluasi Akhir
# # =======================
# preds = model.predict(X_val)
# preds = (preds > 0.5).astype(np.uint8)

# # Hitung IoU
# intersection = np.logical_and(Y_val, preds)
# union = np.logical_or(Y_val, preds)
# iou_score = np.sum(intersection) / np.sum(union)

# # Hitung Dice
# dice_score = (2 * np.sum(intersection)) / (np.sum(Y_val) + np.sum(preds))

# # ✅ Hitung Pixel Accuracy
# pixel_accuracy = np.sum(Y_val == preds) / np.prod(Y_val.shape)

# print("Evaluation Metrics:")
# print(f"IoU Score         : {iou_score:.4f}")
# print(f"Dice Score        : {dice_score:.4f}")
# print(f"Pixel Accuracy    : {pixel_accuracy:.4f}")

# # =================✅ Visualization ==================
# preds = model.predict(X_val[:5])  # 3 Contoh Prediksi

# for i in range(5):
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,3,1); plt.imshow(X_val[i]); plt.title("Image")
#     plt.subplot(1,3,2); plt.imshow(Y_val[i].squeeze(), cmap='gray'); plt.title("Mask")
#     plt.subplot(1,3,3); plt.imshow(preds[i].squeeze()>0.5, cmap='gray'); plt.title("Prediction")
#     plt.show()


# ==========================
# ✅ Evaluation Global
# ==========================
preds = model.predict(X_val)
preds_bin = (preds > 0.5).astype(np.uint8)

intersection = np.logical_and(Y_val, preds_bin)
union = np.logical_or(Y_val, preds_bin)

iou_score = np.sum(intersection) / np.sum(union)
dice_score = (2 * np.sum(intersection)) / (np.sum(Y_val) + np.sum(preds_bin))
pixel_accuracy = np.sum(Y_val == preds_bin) / np.prod(Y_val.shape)

print("\n📊 Evaluation Metrics:")
print(f"IoU Score         : {iou_score:.4f}")
print(f"Dice Score        : {dice_score:.4f}")
print(f"Pixel Accuracy    : {pixel_accuracy:.4f}")

# ==========================
# 🎨 Visualization (5 Samples only on screen)
# ==========================
def overlay_mask(img, mask, color=(255,0,0), alpha=0.5):
    overlay = img.copy()
    overlay[mask.squeeze() == 1] = color
    return (overlay * alpha + img * (1 - alpha)).astype(np.uint8)

num_samples = 5

for i in range(num_samples):

    img = (X_val[i] * 255).astype(np.uint8)
    gt = Y_val[i]
    pr = preds_bin[i]

    overlay_gt = overlay_mask(img, gt,  color=(0,255,0))  # ✅ Hijau = Ground Truth
    overlay_pred = overlay_mask(img, pr, color=(255,0,0)) # ✅ Merah = Prediksi

    fig, ax = plt.subplots(1,5, figsize=(18,4))

    ax[0].imshow(img); ax[0].set_title("Image")
    ax[1].imshow(gt.squeeze(), cmap='gray'); ax[1].set_title("Ground Truth")
    ax[2].imshow(pr.squeeze(), cmap='gray'); ax[2].set_title("Prediction")
    ax[3].imshow(overlay_gt); ax[3].set_title("Overlay GT")
    ax[4].imshow(overlay_pred); ax[4].set_title("Overlay Pred")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()