import cv2
import keras.losses
import numpy as np
import seaborn as sb
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.utils.np_utils import to_categorical

tf.random.set_seed(42)

NUM_IMAGES = 2592
IMG_SIZE = 96
NUM_ROTATIONS = 4

try:
    imgs = np.load('imgs_binary_augmented.npy', allow_pickle=True)
except (OSError, IOError) as e:
    imgs = np.zeros((NUM_IMAGES * NUM_ROTATIONS, IMG_SIZE, IMG_SIZE))
    for i in range(NUM_IMAGES):  # Iterate through each image in the dataset
        print(i)
        img = cv2.imread('star_tracker_dataset/images/stars_{0:03d}.png'.format(i))  # Open the image

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_lp = cv2.GaussianBlur(img_gray, (0, 0), 40)  # Apply Gaussian blur to image
        img_binary = cv2.threshold(img_lp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Binarize image

        old_size = img.shape[:2]

        img_rot_90 = cv2.rotate(img_binary, cv2.ROTATE_90_CLOCKWISE)  # Perform image rotations for augmentation
        img_rot_180 = cv2.rotate(img_binary, cv2.ROTATE_180)
        img_rot_270 = cv2.rotate(img_binary, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ratio = float(IMG_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img_new = cv2.resize(img_binary, (new_size[1], new_size[0]))  # Resize all images to 96x96
        img_rot_90_new = cv2.resize(img_rot_90, (new_size[1], new_size[0]))
        img_rot_180_new = cv2.resize(img_rot_180, (new_size[1], new_size[0]))
        img_rot_270_new = cv2.resize(img_rot_270, (new_size[1], new_size[0]))

        delta_w = IMG_SIZE - new_size[1]
        delta_h = IMG_SIZE - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img_final = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # Pad images
        img_rot_90_final = cv2.copyMakeBorder(img_rot_90_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        img_rot_180_final = cv2.copyMakeBorder(img_rot_180_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        img_rot_270_final = cv2.copyMakeBorder(img_rot_270_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        if i == 0:
            plt.subplot(221), plt.imshow(img_final, cmap='gray')
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.subplot(222), plt.imshow(img_rot_90_final, cmap='gray')
            plt.title('Rotated 90 degrees'), plt.xticks([]), plt.yticks([])
            plt.subplot(223), plt.imshow(img_rot_180_final, cmap='gray')
            plt.title('Rotated 180 degrees'), plt.xticks([]), plt.yticks([])
            plt.subplot(224), plt.imshow(img_rot_270_final, cmap='gray')
            plt.title('Rotated 270 degrees'), plt.xticks([]), plt.yticks([])
            plt.show()
            cv2.imwrite('figures/img_aug.png', img_final)
            cv2.imwrite('figures/img_aug_rot_90.png', img_rot_90_final)
            cv2.imwrite('figures/img_aug_rot_180.png', img_rot_180_final)
            cv2.imwrite('figures/img_aug_rot_270.png', img_rot_270_final)
        imgs[i] = img_final
        imgs[i + NUM_IMAGES] = img_rot_90_final
        imgs[i + 2 * NUM_IMAGES] = img_rot_180_final
        imgs[i + 3 * NUM_IMAGES] = img_rot_270_final
    imgs.dump('imgs_binary_augmented.npy')  # Save the image array

label_lines = open('star_tracker_dataset/labels_augmented.txt', 'r').readlines()
labels = [int(line.split()[0]) for line in label_lines]

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=1 / 10, random_state=42)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=2 / 9, random_state=42)

height = imgs[0].shape[0]
width = imgs[0].shape[1]

x_train = x_train.reshape(-1, height, width, 1)
x_valid = x_valid.reshape(-1, height, width, 1)
x_test = x_test.reshape(-1, height, width, 1)

x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train_hot = to_categorical(y_train)
y_valid_hot = to_categorical(y_valid)
y_test_hot = to_categorical(y_test)

num_classes = len(np.unique(labels))
try:
    cnn = keras.models.load_model('cnn_augmentation')  # Load the CNN if already trained
except (OSError, IOError) as e:
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(height, width, 1)))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D((2, 2), padding='same'))
    cnn.add(Dropout(0.5))
    cnn.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Dropout(0.5))
    cnn.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Dropout(0.5))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='linear'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(num_classes, activation='softmax'))

    cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam_v2.Adam(learning_rate=0.001),
                metrics=['accuracy'])
    cnn.summary()

    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    batch_size = 32
    epochs = 200

    cnn_train = cnn.fit(x_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_valid, y_valid_hot), callbacks=[early])
    cnn.save('cnn_augmentation')

    accuracy = cnn_train.history['accuracy']
    val_accuracy = cnn_train.history['val_accuracy']
    loss = cnn_train.history['loss']
    val_loss = cnn_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

test_eval = cnn.evaluate(x_test, y_test_hot, verbose=0)

print('Test loss: ', test_eval[0])
print('Test accuracy: ', test_eval[1])

labels_verbose = ["North-East", "North-West", "South-East", "South-West"]
y_pred = cnn.predict(x_train)
y_pred = y_pred.argmax(axis=-1)
print("\nCONFUSION MATRIX TRAIN")
confusion_matrix_train = metrics.confusion_matrix(y_train, y_pred)
print(confusion_matrix_train)
print(metrics.classification_report(y_train, y_pred))
sb.heatmap(confusion_matrix_train, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels_verbose,
           yticklabels=labels_verbose)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion matrix for training data')
plt.show()
y_pred_val = cnn.predict(x_valid)
y_pred_val = y_pred_val.argmax(axis=-1)
print("CONFUSION MATRIX VALIDATION")
confusion_matrix_val = metrics.confusion_matrix(y_valid, y_pred_val)
print(confusion_matrix_val)
print(metrics.classification_report(y_valid, y_pred_val))
sb.heatmap(confusion_matrix_val, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels_verbose,
           yticklabels=labels_verbose)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion matrix for validation data')
plt.show()
y_pred_test = cnn.predict(x_test)
y_pred_test = y_pred_test.argmax(axis=-1)
print("CONFUSION MATRIX TEST")
confusion_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)
print(confusion_matrix_test)
print(metrics.classification_report(y_test, y_pred_test))
sb.heatmap(confusion_matrix_test, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels_verbose,
           yticklabels=labels_verbose)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion matrix for test data')
plt.show()

print('Done')
