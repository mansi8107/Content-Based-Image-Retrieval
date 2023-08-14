def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel value is greater than or equal
        # to center pixel values then set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when neighbourhood value of a center
        # pixel value is null i.e. values present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    # set the radius to 1 and consider 8 points for the calculation of LBP of the center pixel
    # this array stores the binary equivalent of the LBP value
    val_ar = []

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # Now, we need to convert binary values to decimal
    power_val = [128, 64, 32, 16, 8, 4, 2, 1]

    value = 0

    for i in range(len(val_ar)):
        value += val_ar[i] * power_val[i]

    return value
