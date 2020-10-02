import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import find_borders, find_countor, RANSAC
import math
import imutils
from planar import BoundingBox

from scipy import signal

"""
0. Delete white frame:
"""


def remove_white_frame(image, top_point):
    h, w = image.shape[:2]
    # from bottom:
    for i in range(h - 150, h):
        for j in range(w):
            image[i][j] = 0

    # from left:
    for i in range(h):
        for j in range(30):
            image[i][j] = 0

    # from top:
    if top_point[1] > 80:
        h_limit = top_point[1]
    else:
        h_limit = 80
    for i in range(h_limit):
        for j in range(w):
            image[i][j] = 0

    # view images:
    cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output1', 600, 600)
    cv2.imshow("output1", image)
    cv2.waitKey(0)
    return image


"""
1. finding width:
"""

"""
    Find an equation straight to the muscle line you tagged
    by the first and the last point we found in find_borders
    return line object
"""


def Finding_Equation_Line(point1, point2):
    point1 = np.array(point1)
    print(point1)
    point2 = np.array(point2)
    print(point2)
    z = np.polyfit(point1, point2, 1)
    m, b = z
    print(m, b)
    line_length = np.poly1d(z)
    # plt.plot(point1, m * point1 + b)
    # plt.show()
    return line_length


"""
    for calculate the nipple line (Width):
    center_point - get the point of the nipple
    slope - the number we get from the function "finding normal" that calculates the slope
    using the equation: y-y1 = m2(x-x1)
    returns poly1d that represents line 
"""


def Finding_Equation_Line_By_Slope_And_Point(slope, center_point):
    x, y, r = center_point
    m = slope
    b = y - slope * x
    line_width = np.poly1d((m, b))
    return line_width


"""
    Finds Intersection Point between line nipple and line muscle.
    return width number
        x = (b2 - b1) / (m1 - m2)
    put x in some equation and got y
"""


def Find_intercept_width_length(line_length, line_width):
    m1, b1 = line_length
    m2, b2 = line_width
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)


"""
    This function gets 2 points of the nipple and the iterception point with the
    length and width equation lines.
    the function returns the width in float.
"""


def Find_Width(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Calculating distance
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


"""
2. finding Length:
"""


# returns the length of the muscle line.
def Finding_Length(poly_top, poly_bottom):
    return (poly_bottom - poly_top)


"""
3. run on all the images: 
Run on all the images, calculate length and width and store the results in two arrays.
After that, calculate for each array its AVG.
"""


# TODO: (Priority = 4)
def calculate_Lengths_and_widths_avg():
    pass


"""
3. Image rotation
"""


def angle_calc(line):
    m1, b = line
    # m2 = 1
    # return math.degrees(math.atan(np.absolute((m2 - m1)/(1+m2*m1))))
    return 90 + math.degrees(math.atan(m1))


# Implement of rotate_bound function:
# def rotate_bound(image, angle):
#     # grab the dimensions of the image and then determine the
#     # centre
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     # perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))

def format_float(num):
    return np.format_float_positional(num, trim='-')


def rotate(image, angle, top_point, nipple_point, eq_line_muscle):
    # Stage 1 - rotation the image by the angle.
    (h, w) = image.shape[:2]
    print("(h, w) before rotate = ", (h, w))
    rotated = imutils.rotate_bound(image, -angle)  # think to be a better option then the ndimage.rotate
    cv2.imwrite("image_after_rotate.png", rotated)

    # Stage 2 - Calculate Distance to shift:
    x, y = top_point
    (h, w) = rotated.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    new_point = find_new_dot(x, y, angle, (cX, cY))
    x_new, y_new = new_point
    # cv2.line(rotated, (int(x_new), int(y_new)), (int(x_new), int(y_new)+1000), color=(255, 0, 255), thickness=20)
    print("(h, w) After rotate = ", (h, w))
    nipple_point = find_new_dot(nipple_point[0], nipple_point[1], angle, (cX, cY))
    x_rotated_nipple, y_rotated_nipple = nipple_point
    cv2.circle(rotated, (int(x_rotated_nipple), int(y_rotated_nipple)), radius=30, color=(255, 0, 0), thickness=20)
    cv2.imwrite("image_after_rotate.png", rotated)
    print("x_rotated_nipple, y_rotated_nipple = ", x_rotated_nipple, y_rotated_nipple)
    # view images:
    cv2.namedWindow('output_rotated', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output_rotated', 600, 600)
    cv2.imshow("output_rotated", rotated)
    cv2.waitKey(0)

    # Stage 3 - shift the image to the left:
    num_rows, num_cols = rotated.shape[:2]
    translation_matrix = np.float32([[1, 0, -x_new], [0, 1, 0]])
    img_translation = cv2.warpAffine(rotated, translation_matrix, ((num_cols, num_rows)))
    (h, w) = img_translation.shape[:2]
    x_shifted_nipple, y_shifted_nipple = x_rotated_nipple - x_new, y_rotated_nipple
    cv2.circle(img_translation, (int(x_shifted_nipple), int(y_shifted_nipple)), radius=30, color=(0, 255, 255),
               thickness=20)
    print("x_shifted_nipple, y_shifted_nipple = ", x_shifted_nipple, y_shifted_nipple)
    nipple_point = (x_shifted_nipple, y_shifted_nipple)
    # cv2.circle(img_translation, (int(x_shifted_nipple), int(y_shifted_nipple)), radius=30, color=(0, 0, 255), thickness=20)
    cv2.imwrite("image_after_shift.png", img_translation)
    # view images:
    cv2.namedWindow('output_rotated', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output_rotated', 600, 600)
    cv2.imshow("output_rotated", rotated)
    cv2.waitKey(0)
    print("(h, w) After shift = ", (h, w))
    return img_translation, nipple_point


"""
4. Compute polygon
"""


# TODO: (Priority = 2) More accurate
def ransac_polyfit(countors, center_point, h, w):
    x_nipple , _ = center_point
    width_box = x_nipple * 1/2  # half of the width .

    bbox = BoundingBox.from_center(center_point, width=width_box, height=2500)  # bounding box from center point

    cx, cy = center_point
    # unzip to x and y arrays:
    bottom_arr_x = np.array([])
    bottom_arr_y = np.array([])
    top_arr_x = np.array([])
    top_arr_y = np.array([])
    countors_test = np.array([])
    for a in countors:
        a_test = np.array([], dtype='int32')
        for b in a:
            # if b[0][0] >= cx + 500:
            #    continue

            v = [b[0][0], b[0][1]]  # current point
            print(v)

            if b[0][1] <= 5 or b[0][1] >= h - 5 or b[0][0] >= w - 5:  # if the contour on the border of the image - ignore it.
                continue

            if not bbox.contains_point(v): #if not in the bound box , continue
                continue

            np.append(a_test, b)
            if b[0][1] <= cy:
                print("enter point")
                top_arr_x = np.append(top_arr_x, b[0][0])
                top_arr_y = np.append(top_arr_y, -1 * b[0][1])  # multiply with -1 only for desmos view!!
            else:
                bottom_arr_x = np.append(bottom_arr_x, b[0][0])
                bottom_arr_y = np.append(bottom_arr_y, -1 * b[0][1])  # multiply with -1 only for desmos view!!
        np.append(countors_test, a_test)

    print("TOP1")
    coeff_top = RANSAC.quadratic_ransac_curve_fit("Upper Polynomial", top_arr_x, top_arr_y)
    print("BOTTOM1")
    coeff_bottom = RANSAC.quadratic_ransac_curve_fit("Lower Polynomial", bottom_arr_x, bottom_arr_y)
    return (coeff_top, coeff_bottom)


def derivative(poly):
    derived_coeffs = np.array([])
    exponent = len(poly) - 1
    for i in range(len(poly) - 1):
        derived_coeffs = np.append(derived_coeffs, poly[i] * exponent)
        exponent -= 1
    return derived_coeffs


def f(x):
    exp = (math.pow(x[0], 2) + math.pow(x[1], 2)) * -1
    return math.exp(exp) * math.cos(x[0] * x[1]) * math.sin(x[0] * x[1])


"""
This function transform point in rotated image. 
"""


def find_new_dot(x, y, angle, center):
    angle_rad = math.radians(angle)
    cX, cY = center
    print("angle_rad = ", angle_rad)
    x_new = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_new = -(x - cX) * np.sin(angle_rad) + (y - cY) * np.cos(angle_rad) + cY
    return (x_new, y_new)


def main():
    # Read tagged and source images:
    tagged = cv2.imread("images\\Mass-Test_P_00358_RIGHT_MLO_tagged.png")
    source = cv2.imread("images\\Mass-Test_P_00358_RIGHT_MLO.png")
    # tagged = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC_Tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC.png")
    h, w = tagged.shape[:2]
    print("h, w start = ", h, w)
    # Find line of muscle and nipple
    top_point, buttom_point = find_borders.findLine(tagged)

    # Delete white frame
    source = remove_white_frame(source, top_point)

    # Find Equation line and angle for rotation
    eq_line_muscle = Finding_Equation_Line(top_point, buttom_point)
    print("eq_line_muscle = ", eq_line_muscle)
    angle = angle_calc(eq_line_muscle)
    print("angle = ", angle)
    _, nipple_point = find_borders.findCircle(tagged)
    print("nipple_point start = ", nipple_point)

    if angle < 90:
        source_rotated, nipple_point = rotate(source, angle, top_point, nipple_point, eq_line_muscle)
        source = source_rotated

    # RANSAC:
    # TODO: (Priority = 1) More accurate contours, with minimum noise + Check on several images.
    countors = find_countor.getEdgeImage(source)
    poly_top, poly_bottom = ransac_polyfit(countors, nipple_point, h, w)

    cv2.circle(source, (int(0), int(poly_top[0])), radius=30, color=(0, 255, 255), thickness=20)
    cv2.circle(source, (int(0), int(poly_bottom[0])), radius=30, color=(0, 255, 255), thickness=20)

    bbox = BoundingBox.from_center(nipple_point, width=1000, height=2500)  # bounding box

    cv2.rectangle(source, (1497-500 , 3727-1250), (1497+500, 3727+1250),(255, 0, 0),20)
    # view images:
    cv2.namedWindow('test_image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test_image', 600, 600)
    cv2.imshow("test_image", source)
    cv2.waitKey(0)

    # calculate new length of muscle:
    muscle_length = Finding_Length(poly_top, poly_bottom)

    # TODO: (Priority = 3) Check if true, after get to the best result.
    # calculate width:
    width_length = nipple_point[0]
    width_equation = np.array(nipple_point[1])
    width_iter_muscle = (0, nipple_point[1])  # The intercetion point between the muscle line to the width line.


if __name__ == '__main__':
    main()
