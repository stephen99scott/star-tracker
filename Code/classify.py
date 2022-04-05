import cv2
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

NUM_IMAGES = 2592
NUM_CONTOURS = 5
NUM_FEATURES = 16

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
            img_cnt = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
            cv2.imwrite('figures/all_contours.png', img_cnt)
            cv2.imshow('Contours', img_cnt)
            cv2.waitKey(0)
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        idxs = areas.argsort()[::-1]
        biggest_contour = contours[idxs[0]]
        contour_centers = np.zeros((NUM_CONTOURS + 1, 2))
        (x, y), radius = cv2.minEnclosingCircle(biggest_contour)
        features[i][0] = radius
        contour_centers[0] = [x, y]

        mask = np.zeros_like(img_binary)
        cx, cy = int(x), int(y)
        cv2.circle(mask, (cx, cy), 500, 255, -1)
        masked = cv2.bitwise_and(img_binary, img_binary, mask=mask)
        if i == 0:
            plt.subplot(121), plt.imshow(mask, cmap='gray')
            plt.title('Mask'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(masked, cmap='gray')
            plt.title('Masked'), plt.xticks([]), plt.yticks([])
            plt.show()
        pixels = cv2.countNonZero(masked)
        features[i][1] = pixels

        contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if i == 0:  # Visualize some data if this is the first iteration
            img_cnt = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
            cv2.imwrite('figures/local_contours.png', img_cnt)
            cv2.imshow('Contours', img_cnt)
            cv2.waitKey(0)
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        idxs = areas.argsort()[::-1][:NUM_CONTOURS]
        biggest_contours = [contours[i] for i in idxs]
        contour_centers = np.zeros((NUM_CONTOURS, 2))
        for j in range(1, NUM_CONTOURS):
            if j >= len(biggest_contours):
                contour_centers[j] = contour_centers[0]
                continue
            (x, y), radius = cv2.minEnclosingCircle(biggest_contours[j])
            features[i][j + 1] = radius
            contour_centers[j] = [x, y]
        contour_centers_norms = np.linalg.norm(contour_centers - contour_centers[:, None], axis=-1)
        s = 0
        for j in range(1, NUM_CONTOURS):
            features[i][NUM_CONTOURS + 1 + s:2 * NUM_CONTOURS + s + 1 - j] = contour_centers_norms[j:NUM_CONTOURS,
                                                                             j - 1]
            s += NUM_CONTOURS - j

    features.dump('features.npy')

label_lines = open('star_tracker_dataset/labels.txt', 'r').readlines()
labels = [int(line.split()[0]) for line in label_lines]

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1 / 5, random_state=42)

x_train_normalized, norm = preprocessing.normalize(x_train, axis=0, return_norm=True)

covariance_matrix = np.cov(x_train_normalized.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
projection = (eigenvectors.T[:][:3]).T
pca_train = x_train_normalized.dot(projection)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], c=y_train)
plt.show()

# parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100, 1000],
#               'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# clf = GridSearchCV(SVC(), parameters, verbose=3)
# clf.fit(x_train_normalized, y_train)
# print(clf.best_score_)
# print(clf.best_estimator_)
# print(clf.best_params_)
labels_verbose = ["North-East", "North-West", "South-East", "South-West"]
model = SVC(kernel='rbf', gamma=1000, C=100)
model.fit(x_train_normalized, y_train)
y_pred = model.predict(x_train_normalized)
print("\nCONFUSION MATRIX TRAIN")
confusion_matrix_train = metrics.confusion_matrix(y_train, y_pred)
print(confusion_matrix_train)
print(metrics.classification_report(y_train, y_pred))
unique, counts = np.unique(labels, return_counts=True)
sb.heatmap(confusion_matrix_train, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels_verbose,
           yticklabels=labels_verbose)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion matrix for training data')
plt.show()
x_test_normalized = np.divide(x_test, norm)
y_pred_test = model.predict(x_test_normalized)
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
