import cv2
import numpy as np
import os

from lbp import lbp_calculated_pixel
from colorMoments import mean, standard_deviation, skewness
import pandas as pd
from collections import Counter


def convert_lbp(img_rgb):
    height, width, _ = img_rgb.shape

    # We need to convert RGB image into gray one because gray image has one channel only.
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Create a numpy array as the same height and width of RGB image
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp


def convert_img(img_rgb):
    # resizing or standardising all the images to a size of (400 x 400)
    img_rgb = cv2.resize(img_rgb, dsize=(400, 400))

    img_lbp = convert_lbp(img_rgb)

    img_lbp_flatten = img_lbp.flatten()

    return np.array(img_lbp_flatten)


def load_images_from_folder(folder):
    all_images = []
    for i in range(1000):
        image = cv2.imread(os.path.join(folder, f"{i}.jpg"))
        if image is not None:
            all_images.append(image)
            print(f"Loading done for: {i}.jpg")
    return all_images


def create_histogram(fv):
    frequency_fv = Counter(fv)

    hist = []
    for i in range(256):
        if i in frequency_fv.keys():
            hist.append(frequency_fv[i])
        else:
            hist.append(0)
    return hist


def extract_features(image):
    # 1-D vector of image consisting of 400*400 values of pixels converted to LBP value
    image_lbp = convert_img(image)

    image_lbp_histogram = create_histogram(image_lbp)

    # histogram created from the lbp values is the feature vector of the image
    lbp_feature_vector = image_lbp_histogram / np.sum(image_lbp_histogram)

    # calculating mean, variance and skewness of each channel (HSV)
    # Converting the RGB image to HSV for Colour Moments
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_mean, s_mean, v_mean = mean(img_hsv)
    h_deviation, s_deviation, v_deviation = standard_deviation(img_hsv)
    h_skewness, s_skewness, v_skewness = skewness(img_hsv)

    # Colour Moments feature vectors consisting of nine values
    cm_feature_vector = [h_mean, h_deviation, h_skewness, s_mean, s_deviation, s_skewness, v_mean, v_deviation,
                         v_skewness]

    return lbp_feature_vector, cm_feature_vector


def create_features():
    print("Loading images started ======================================================")
    all_images = load_images_from_folder('images')
    print("Loading images finished =====================================================")

    # create feature vector of all the database images that contains LBP and CM both,
    # order -> LBP histogram features and then Red(3), Green(3), B(3)
    all_images_feature_vector = []

    for index, image in enumerate(all_images):
        print("Calculating for: ", index)
        lbp_feature_vector, cm_feature_vector = extract_features(image)

        # concatenating the lbp and cm feature vector to create final feature vector
        # that would be used for similarity calculations
        feature_vector_combined = np.concatenate((lbp_feature_vector, cm_feature_vector), axis=0)
        all_images_feature_vector.append(feature_vector_combined)
        print("Done with: ", index)

    all_images_feature_vector_array = np.asarray(all_images_feature_vector)
    pd.DataFrame(all_images_feature_vector_array).to_csv('database_features.csv')


if __name__ == '__main__':
    create_features()
