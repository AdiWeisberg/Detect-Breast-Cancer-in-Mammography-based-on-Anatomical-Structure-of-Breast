import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import find_borders, find_countor, RANSAC
import math
import imutils
from planar import BoundingBox
from collections import Counter

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
    # cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output1', 600, 600)
    # cv2.imshow("output1", image)
    # cv2.waitKey(0)
    return image


def remove_white_frame_norotate(image, top_point, buttom_point):
    h, w = image.shape[:2]

    # from left:
    # for i in range(0, top_point[1]-10):
    #     for j in range(30):
    #         image[i][j] = 0
    #
    # for i in range(buttom_point[1]+10, h):
    #     for j in range(30):
    #         image[i][j] = 0

    # from bottom:
    for i in range(h - 150, h):
        for j in range(w):
            image[i][j] = 0

    # from top:
    if top_point[1] > 80:
        h_limit = top_point[1]
    else:
        h_limit = 80
    for i in range(h_limit):
        for j in range(w):
            image[i][j] = 0


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
    # cv2.circle(rotated, (int(x_rotated_nipple), int(y_rotated_nipple)), radius=30, color=(255, 0, 0), thickness=20)
    cv2.imwrite("image_after_rotate.png", rotated)
    print("x_rotated_nipple, y_rotated_nipple = ", x_rotated_nipple, y_rotated_nipple)
    # view images:
    # cv2.namedWindow('output_rotated', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output_rotated', 600, 600)
    # cv2.imshow("output_rotated", rotated)
    # cv2.waitKey(0)

    # Stage 3 - shift the image to the left:
    num_rows, num_cols = rotated.shape[:2]
    translation_matrix = np.float32([[1, 0, -x_new], [0, 1, 0]])
    img_translation = cv2.warpAffine(rotated, translation_matrix, ((num_cols, num_rows)))
    (h, w) = img_translation.shape[:2]
    x_shifted_nipple, y_shifted_nipple = x_rotated_nipple - x_new, y_rotated_nipple
    # cv2.circle(img_translation, (int(x_shifted_nipple), int(y_shifted_nipple)), radius=30, color=(0, 255, 255),thickness=20)
    print("x_shifted_nipple, y_shifted_nipple = ", x_shifted_nipple, y_shifted_nipple)
    nipple_point = (x_shifted_nipple, y_shifted_nipple)
    # cv2.circle(img_translation, (int(x_shifted_nipple), int(y_shifted_nipple)), radius=30, color=(0, 0, 255), thickness=20)
    cv2.imwrite("image_after_shift.png", img_translation)
    # view images:
    # cv2.namedWindow('output_rotated', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output_rotated', 600, 600)
    # cv2.imshow("output_rotated", rotated)
    # cv2.waitKey(0)
    print("(h, w) After shift = ", (h, w))
    return img_translation, nipple_point


"""
4. Compute polygon
"""


# TODO: (Priority = 2) More accurate
def ransac_polyfit(countors, center_point, h, w, source):
    x_nipple, _ = center_point
    width_box = x_nipple * 1 / 2  # half of the width .

    # bbox = BoundingBox.from_center(center_point, width=w, height=2500)  # bounding box from center point
    # bbox = BoundingBox([(), ()])  # bounding box from center point
    # temp_image = np.copy(source)
    # cv2.rectangle(temp_image, (1497 - 500, 3727 - 1250), (1497 + 500, 3727 + 1250), (255, 0, 0), 20)
    # view images:
    # cv2.namedWindow('test_image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('test_image', 600, 600)
    # cv2.imshow("test_image", source)
    # cv2.waitKey(0)

    cx, cy = center_point
    # unzip to x and y arrays:
    bottom_arr_x = np.array([])
    bottom_arr_y = np.array([])
    top_arr_x = np.array([])
    top_arr_y = np.array([])
    poly_top = []
    poly_bottom = []
    # countors = sorted(countors, key=cv2.contourArea, reverse=True)[:1]
    original_image = np.copy(source)
    countors = sorted(countors, key=cv2.contourArea, reverse=True)[:1]
    for c in countors:
        cv2.drawContours(original_image, [c], -1, (255, 0, 0), 10)
        cv2.namedWindow('Contours_By_Area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours_By_Area', 600, 600)
        cv2.imshow('Contours_By_Area', original_image)
        cv2.waitKey(0)

    for a in countors:
        print("~~~~~~~~~~~~")
        print(a)
    ############
    image_with_contours_1 = np.copy(source)
    image_with_contours = cv2.drawContours(image_with_contours_1, countors, -1, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imwrite("image_with_contours_1.png", image_with_contours)
    #########
    for a in countors:
        for b in a:
            # if b[0][0] > cx + 200:
            #     continue
            # if b[0][1] <= 5 or b[0][1] >= h - 5 or b[0][0] >= w - 5:  # if the contour on the border of the image - ignore it.
            #     continue

            # if not bbox.contains_point(v): #if not in the bound box , continue
            #    continue

            if b[0][1] <= cy:
                poly_top.append(b[0])
                top_arr_x = np.append(top_arr_x, b[0][0])
                top_arr_y = np.append(top_arr_y, -1 * b[0][1])  # multiply with -1 only for desmos view!!
            else:
                poly_bottom.append(b[0])
                bottom_arr_x = np.append(bottom_arr_x, b[0][0])
                bottom_arr_y = np.append(bottom_arr_y, -1 * b[0][1])  # multiply with -1 only for desmos view!!

    top_dot = gradient_calc(poly_top, center_point, "Top")
    bottom_dot = gradient_calc(poly_bottom, center_point, "Bottom")
    print(" top_muscle = ", top_dot)
    print(" bottom_muscle = ", bottom_dot)

    print("TOP1")
    coeff_top = RANSAC.quadratic_ransac_curve_fit("Upper Polynomial", top_arr_x, top_arr_y)
    print("BOTTOM1")
    coeff_bottom = RANSAC.quadratic_ransac_curve_fit("Lower Polynomial", bottom_arr_x, bottom_arr_y)
    return (coeff_top, coeff_bottom)


def gradient_calc(poly, nipple_point, flag):
    xy = (-1, -1)
    x_old = nipple_point[0]
    y_old = nipple_point[1]
    counter_positive = -1
    counter_negative = -1
    last_ten = []
    res = (-1, -1)
    list_of_res = []
    res_counter = dict()
    avg_ten_m = -100
    cout_signs = 0
    old_ten_avg = -100
    n = 30
    if flag == "Bottom":
        print("~~~~~~~~~~~~~~ Bottom ~~~~~~~~~~~~~~")
        counter_to_ten = 0
        print(" poly[0] = ", poly[0])
        m = (poly[0][1] - nipple_point[1]) / (poly[0][0] - nipple_point[0])
        x_old = nipple_point[0]
        y_old = nipple_point[1]
        m_old = m
        first_time = False
        for i, points in enumerate(poly[int(len(poly) * 0.5)::50]):
            x, y = points[0], points[1]
            if x_old == x or y_old == y:
                continue

            if np.isinf(m):
                m = m_old
            print(" gradient = ", (poly[i][1] - y_old) / (poly[i][0] - x_old))
            m = (poly[i][1] - y_old) / (poly[i][0] - x_old)
            tetha_curr = math.degrees(math.atan(m))
            if counter_to_ten < n - 1:
                last_ten.append([m, (x, y)])
                counter_to_ten += 1
                print(" current point m = ", m)
                print(" tetha_curr = ", tetha_curr)
                print(" x ,y = ", (x, y))
            elif counter_to_ten == n - 1:
                last_ten.append([m, (x, y)])
                only_m = []
                for j in range(0, len(last_ten)):
                    only_m.append(last_ten[j][0])
                avg_ten_m = sum(only_m) / n
                cout_signs = sum(1 for c in last_ten if c[0] > 0)  # count 3 pluses +++
                index_max_m = np.argmax(only_m)
                res = last_ten[index_max_m][1]
                list_of_res.append(res)
                y_max = sorted(list_of_res, key=lambda x: x[1], reverse=True)
                print(" y_max = ", y_max)
                res_counter = Counter(list_of_res)
                if -1 < avg_ten_m < 0 and x_old - x <= 20 and avg_ten_m > old_ten_avg and y_old < y and abs(
                        y_old - y) < 100:
                    if not first_time:
                        list_of_res.append(last_ten[-1][1])
                        first_time = True
                    print("Point found!!")
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[0][0])
                    print(" tetha_curr = ", tetha_curr)
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", res_counter.most_common())
                    # break
                else:
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[-1][0])
                    print(" tetha_curr = ", tetha_curr)
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", res_counter.most_common())
                last_ten.pop(0)
                old_ten_avg = avg_ten_m

            x_old = x
            y_old = y
            m_old = m
            tetha = tetha_curr
        xy = res_counter.most_common(1)[0][0]

    if flag == "Top":
        print("~~~~~~~~~~~~~~ Top ~~~~~~~~~~~~~~")
        counter_to_ten = 0
        m = (poly[-1][1] - nipple_point[1]) / (poly[-1][0] - nipple_point[0])
        tetha = math.degrees(math.atan(m))
        # print(" tetha = ", tetha)
        m_old = m
        x_old = nipple_point[0]
        y_old = nipple_point[1]
        tetha_curr_old = 0
        first_time = False
        for i, points in enumerate(poly[-int(len(poly) * 0.5)::-50]):
            # if (x_old - x) == 50 or (abs(x_old - x) > 200):
            #     continue
            x, y = points[0], points[1]
            if x_old == x or y_old == y:
                continue
            if np.isinf(m):
                m = m_old
            print(" gradient = ", (poly[i][1] - y_old) / (poly[i][0] - x_old))
            m = (poly[i][1] - y_old) / (poly[i][0] - x_old)
            tetha_curr = math.degrees(math.atan(m))
            if counter_to_ten < n - 1:
                last_ten.append([m, (x, y)])
                counter_to_ten += 1
                print(" current point m = ", m)
                print(" tetha_curr = ", tetha_curr)
                print(" x ,y = ", (x, y))
            elif counter_to_ten == n - 1:
                last_ten.append([m, (x, y), tetha_curr])
                only_m = []
                for j in range(0, len(last_ten)):
                    only_m.append(last_ten[j][0])
                avg_ten_m = sum(only_m) / n
                count_signs = sum(1 for c in last_ten if c[0] < 0)  # count 3 minuses or more ---
                index_min_m = np.argmin(only_m)
                res = last_ten[index_min_m][1]
                list_of_res.append(res)
                y_min = sorted(list_of_res, key=lambda x: x[1])
                print(" y_min = ", y_min)
                res_counter = Counter(list_of_res)
                if avg_ten_m > old_ten_avg and count_signs >= 5 and int(tetha_curr) != int(tetha_curr_old):
                    if not first_time:
                        list_of_res.append(last_ten[-1][1])
                        first_time = True
                    print("Point found!!")
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[-1][0])
                    print(" tetha_curr = ", tetha_curr)
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", res_counter.most_common())
                    # break
                else:
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[-1][0])
                    print(" tetha_curr = ", tetha_curr)
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", res_counter.most_common())
                last_ten.pop(0)
                old_ten_avg = avg_ten_m
            x_old = x
            y_old = y
            m_old = m
            tetha_curr_old = tetha_curr
        xy = y_min[0]
    print(" point is ", xy)
    return xy


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

"""
3. run on all the images: 
Run on all the images, calculate length and width and store the results in two arrays.
After that, calculate for each array its AVG.
"""


def sum_calc(size_breast_list):
    widths = 0
    lengths = 0

    for i in size_breast_list:
        width, length = i
        widths += width
        lengths += length

    size = length(size_breast_list)
    return widths, lengths, size


"""
3. ratio between the width and length avg

"""


def ratio_calc(list_of_wh):
    widths = 0
    lengths = 0
    sum = 0

    for i in list_of_wh:
        w, h, size = i
        widths += w
        lengths += h
        sum += size

    ratio = (widths / sum) / (lengths / sum)
    return ratio


# crop from the image only the brest .
# y top - is the top of the picture , so its number smaller then the y_bottom because of the coordinates.
def crop_breast_from_image(x_nipple, y_top, y_bottom, image):
    cropped_image = image[y_top:y_bottom, 0:x_nipple]
    return cropped_image


# calculates the change of image size by ratio info.
# the ratio calculated within the function "ratio_calc"
def change_image_by_ratio(xy_breast_list, list_after_ration, ratio_avg):
    for i in xy_breast_list:
        w, h = i
        add_for_x = ratio_avg * h - w
        w = w + add_for_x
        list_after_ration.insert(i, (w, h))
    return list_after_ration


# checks the max x and max y in list.
def max_image_size(list_after_ration, x_max=0, y_max=0):
    for i in list_after_ration:
        x, y = i
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    return x_max, y_max


# sizes of max image - x_max,y_max
def change_image_size(image, path, image_name, list_of_sizes, x_max, y_max):
    # First resize image by ratio
    w, h = list_of_sizes[image_name]
    dim = (w, h)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Second paste image on blank image with max size
    blank_image = np.zeros((y_max, x_max, 3), np.uint8)
    blank_image[:image.shape[0], :image.shape[1]] = image


    cv2.imshow("resized_image", resized_image)
    cv2.waitKey(0)
    cv2.imshow("paste_image", blank_image)
    cv2.waitKey(0)

    cv2.imwrite(image_name, blank_image, path)

def main():
    # Read tagged and source images:
    # tagged = cv2.imread("images\\Mass-Test_P_01140_LEFT_MLO_tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_01140_LEFT_MLO1-1.png")
    # tagged = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC_Tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC.png")
    # tagged = cv2.imread("images\\Mass-Test_P_01348_LEFT_MLO_tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_01348_LEFT_MLO.png")
    # tagged = cv2.imread("images\\Mass-Test_P_01719_RIGHT_CC_tagged.png") - not accurate!
    # source = cv2.imread("images\\Mass-Test_P_01719_RIGHT_CC.png")
    tagged = cv2.imread("images\\Calc-Test_P_01883_RIGHT_MLO_tagged.png")
    source = cv2.imread("images\\Calc-Test_P_01883_RIGHT_MLO.png")
    h, w = tagged.shape[:2]
    print("h, w start = ", h, w)
    # Find line of muscle and nipple
    top_point, buttom_point = find_borders.findLine(tagged)

    # Find Equation line and angle for rotation
    eq_line_muscle = Finding_Equation_Line(top_point, buttom_point)
    print("eq_line_muscle = ", eq_line_muscle)
    angle = angle_calc(eq_line_muscle)
    print("angle = ", angle)
    _, nipple_point = find_borders.findCircle(tagged)
    print("nipple_point start = ", nipple_point)

    #if angle > 90:
    #    source = remove_white_frame_norotate(source, top_point, buttom_point)

    if angle < 90:
        # Delete white frame
        source = remove_white_frame(source, top_point)

        source_rotated, nipple_point = rotate(source, angle, top_point, nipple_point, eq_line_muscle)
        source = source_rotated

    # RANSAC:
    # TODO: (Priority = 1) More accurate contours, with minimum noise + Check on several images.
    countors = find_countor.getEdgeImage(source)
    poly_top, poly_bottom = ransac_polyfit(countors, nipple_point, h, w, source)

    # deriv_top = derivative(poly_top)
    # deriv_bottom = derivative(poly_bottom)
    # print('Equation: {0:.20f} + {1:.20f}x + {2:.20f}x^2 + {3:.20f}x^3 + {4:.20f}x^4'.format(deriv_top[0], deriv_top[1], deriv_top[2], deriv_top[3], deriv_top[4]))
    # print('Equation: {0:.20f} + {1:.20f}x + {2:.20f}x^2 + {3:.20f}x^3 + {4:.20f}x^4'.format(deriv_bottom[0], deriv_bottom[1], deriv_bottom[2], deriv_bottom[3], deriv_bottom[4]))

    # cv2.circle(source, (int(0), int(poly_top[0])), radius=30, color=(0, 255, 255), thickness=20)
    # cv2.circle(source, (int(0), int(poly_bottom[0])), radius=30, color=(0, 255, 255), thickness=20)

    # bbox = BoundingBox.from_center(nipple_point, width=1000, height=2500)  # bounding box

    # cv2.rectangle(source, (1497-500 , 3727-1250), (1497+500, 3727+1250),(255, 0, 0),20)
    # view images:
    # cv2.namedWindow('test_image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('test_image', 600, 600)
    # cv2.imshow("test_image", source)
    # cv2.waitKey(0)

    # calculate new length of muscle:
    muscle_length = Finding_Length(poly_top, poly_bottom)

    # TODO: (Priority = 3) Check if true, after get to the best result.
    # calculate width:
    width_length = nipple_point[0]
    width_equation = np.array(nipple_point[1])
    width_iter_muscle = (0, nipple_point[1])  # The intercetion point between the muscle line to the width line.


if __name__ == '__main__':
    main()
