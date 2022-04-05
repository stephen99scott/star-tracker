import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns

NUM_IMAGES = 2592
NUM_CONTOURS = 10
NUM_FEATURES = 68

try:
    features = np.load('features.npy', allow_pickle=True)
except (OSError, IOError) as e:
    features = np.zeros((NUM_IMAGES, NUM_FEATURES))

    for i in range(NUM_IMAGES):  # Iterate through each image in the dataset
        print(i)
        img = cv2.imread('star_tracker_dataset/images/stars_{0:03d}.png'.format(i))  # Open the image

        # PREPROCESSING
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        img_lp = cv2.GaussianBlur(img_gray, (0, 0), 2)  # Apply Gaussian to image
        img_binary = cv2.threshold(img_lp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Binarize image

        # FEATURE EXTRACTION
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        if i == 0:  # Visualize some data if this is the first iteration
            plt.subplot(121), plt.imshow(img_lp, cmap='gray')
            plt.title('Gauss'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img_binary, cmap='gray')
            plt.title('Binary'), plt.xticks([]), plt.yticks([])
            plt.show()
            img_cnt = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
            cv2.imshow('Contours', img_cnt)
            cv2.waitKey(0)
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        idxs = areas.argsort()[::-1][:NUM_CONTOURS]
        biggest_contours = [contours[i] for i in idxs]
        contour_centers = np.zeros((NUM_CONTOURS, 2))
        for j in range(NUM_CONTOURS):
            (x, y), radius = cv2.minEnclosingCircle(biggest_contours[j])
            features[i][j] = radius
            contour_centers[j] = [x, y]
        contour_centers_norms = np.linalg.norm(contour_centers - contour_centers[:, None], axis=-1)
        s = 0
        for j in range(1, NUM_CONTOURS):
            features[i][NUM_CONTOURS + s:NUM_CONTOURS + s + NUM_CONTOURS - j] = contour_centers_norms[j:NUM_CONTOURS,
                                                                                j - 1]
            s += NUM_CONTOURS - j

        mask = np.zeros_like(img_gray)
        cx, cy = img_binary.shape[1] // 2, img_binary.shape[0] // 2
        cx, cy = int(cx), int(cy)
        max_radius = img_gray.shape[1] // 2
        min_radius = 100
        radius = min_radius
        prev_pixels = 0
        feature_idx = NUM_CONTOURS + s
        while radius < max_radius:
            cv2.circle(mask, (cx, cy), radius, 1, -1)
            masked = cv2.bitwise_and(img_binary, img_binary, mask=mask)
            pixels = cv2.countNonZero(masked)
            features[i][feature_idx] = pixels - prev_pixels
            radius += 100
            feature_idx += 1
            prev_pixels = pixels
        features[i][feature_idx] = cv2.countNonZero(img_binary) - prev_pixels
    features.dump('features.npy')

label_lines = open('star_tracker_dataset/labels.txt', 'r').readlines()
labels = [int(line.split()[0]) for line in label_lines]

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1 / 3, random_state=42)

x_train_normalized, norm = preprocessing.normalize(x_train, axis=0, return_norm=True)

covariance_matrix = np.cov(x_train_normalized.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
projection = (eigenvectors.T[:][:20]).T
pca_train = x_train_normalized.dot(projection)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], c=y_train)
plt.show()

# parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100, 1000],
#               'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# clf = GridSearchCV(SVC(), parameters, verbose=3)
# clf.fit(pca_train, y_train)
# print(clf.best_score_)
# print(clf.best_estimator_)
# print(clf.best_params_)
model = SVC(kernel='linear', gamma=0.001, C=0.01)
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
print("\nCONFUSION MATRIX TRAIN")
confusion_matrix_train = metrics.confusion_matrix(y_train, y_pred)
print(confusion_matrix_train)
print(metrics.classification_report(y_train, y_pred))
unique, counts = np.unique(labels, return_counts=True)
# sns.heatmap(confusion_matrix_train, square=True, annot=True, fmt='d', cbar=False, xticklabels=unique,
#             yticklabels=unique)
# plt.xlabel('Prediction')
# plt.ylabel('Actual')
# plt.title('Confusion matrix for training data')
# plt.show()
y_pred_test = model.predict(x_test)
print("CONFUSION MATRIX TEST")
confusion_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)
print(confusion_matrix_test)
print(metrics.classification_report(y_test, y_pred_test))
# sns.heatmap(confusion_matrix_test, square=True, annot=True, fmt='d', cbar=False, xticklabels=unique,
#             yticklabels=unique)
# plt.xlabel('Prediction')
# plt.ylabel('Actual')
# plt.title('Confusion matrix for test data')
# plt.show()

print('Done')
