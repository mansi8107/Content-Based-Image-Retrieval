import numpy as np


# splitting the different channels of the RGB image
def create_feature_matrix_for_channel(img_hsv):
    feature_h_matrix = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
    feature_s_matrix = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
    feature_v_matrix = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))

    for k in range(0, 3):
        for i in range(img_hsv.shape[0]):
            for j in range(img_hsv.shape[1]):
                if k == 0:
                    feature_h_matrix[i][j] = int(img_hsv[i, j, 0])
                elif k == 1:
                    feature_s_matrix[i][j] = int(img_hsv[i, j, 1])
                else:
                    feature_v_matrix[i][j] = int(img_hsv[i, j, 2])

    return feature_h_matrix, feature_s_matrix, feature_v_matrix


# calculating the mean (first moment) value for each of the channel
def mean(img_hsv):
    feature_h_matrix, feature_s_matrix, feature_v_matrix = create_feature_matrix_for_channel(img_hsv)

    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

    mean_h = np.sum(feature_h_matrix) / total_pixels
    mean_s = np.sum(feature_s_matrix) / total_pixels
    mean_v = np.sum(feature_v_matrix) / total_pixels

    return mean_h, mean_s, mean_v


def calculate_deviation_channel(feature_matrix, mean_channel, total_pixels):
    deviation = 0

    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            deviation += np.square(feature_matrix[i][j] - mean_channel)

    return np.sqrt(deviation / total_pixels)


# calculating the standard deviation (second moment) value for each of the channel
def standard_deviation(img_hsv):
    feature_h_matrix, feature_s_matrix, feature_v_matrix = create_feature_matrix_for_channel(img_hsv)
    mean_h, mean_s, mean_v = mean(img_hsv)

    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

    h_deviation = calculate_deviation_channel(feature_h_matrix, mean_h, total_pixels)
    s_deviation = calculate_deviation_channel(feature_s_matrix, mean_s, total_pixels)
    v_deviation = calculate_deviation_channel(feature_v_matrix, mean_v, total_pixels)

    return h_deviation, s_deviation, v_deviation


def calculate_skewness_channel(feature_matrix, mean_channel, total_pixels):
    skew = 0

    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            skew += (feature_matrix[i][j] - mean_channel) ** 3

    return np.cbrt(skew / total_pixels)


# calculating the skewness (third moment) value for each of the channel
def skewness(img_hsv):
    feature_h_matrix, feature_s_matrix, feature_v_matrix = create_feature_matrix_for_channel(img_hsv)
    mean_h, mean_s, mean_v = mean(img_hsv)

    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

    h_skewness = calculate_skewness_channel(feature_h_matrix, mean_h, total_pixels)
    s_skewness = calculate_skewness_channel(feature_s_matrix, mean_s, total_pixels)
    v_skewness = calculate_skewness_channel(feature_v_matrix, mean_v, total_pixels)

    return h_skewness, s_skewness, v_skewness


def calculate_color_moment(db_cm, q_cm):
    d_mom = 0

    for j in range(3):
        d_mom += (abs(db_cm[j][0] - q_cm[j][0]) / (abs(db_cm[j][0]) + abs(q_cm[j][0])))  # added mean
        d_mom += (abs(db_cm[j][1] - q_cm[j][1]) / (
                abs(db_cm[j][1]) + abs(q_cm[j][1])))  # added standard deviation
        d_mom += (abs(db_cm[j][2] - q_cm[j][2]) / (abs(db_cm[j][2]) + abs(q_cm[j][2])))  # added skewness

    return d_mom

