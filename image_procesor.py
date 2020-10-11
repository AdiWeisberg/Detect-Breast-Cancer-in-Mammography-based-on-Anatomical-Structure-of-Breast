import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import find_borders, find_countor, RANSAC
import math
import imutils
import pydicom
from collections import Counter

# from planar import BoundingBox
# from scipy import signal

"""
0. Delete white frame:
"""

"""
    This function removes white frame from images that needed to be rotated from all sides. 
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

    return image


"""
    This function removes white frame from images has straight muscle line and do not need to be rotated. In this case - the line that limit the breast 
    should not be blacken, cause if so the biggest contour won't include the breast. 
"""


def remove_white_frame_norotate(image, top_point, buttom_point):
    h, w = image.shape[:2]
    # from bottom:
    if abs(h - buttom_point[1]) > 200:
        h_limit_bottom = buttom_point[1] + 200
    else:
        h_limit_bottom = buttom_point[1] + 100
    for i in range(h_limit_bottom, h):
        for j in range(w):
            image[i][j] = 0

    # from top:
    if abs(0 - top_point[1]) > 50:
        h_limit_top = top_point[1] - 50
    else:
        h_limit_top = top_point[1] - 20
    for i in range(h_limit_top):
        for j in range(w):
            image[i][j] = 0

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
def Finding_Length(top_muscle, bottom_muscle):
    x1, y1 = bottom_muscle
    x2, y2 = top_muscle
    return math.hypot(x2 - x1, y2 - y1)


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


"""
    This function gets a number and returns the representation of it without notation (e signs). 
"""


def format_float(num):
    return np.format_float_positional(num, trim='-')


"""
    This function gets an image and rotate it so the muscle line will be straight and stick to the left side of the image. 
    We use imutils.rotate_bound to insure that the image won't be cropped. 
    The function returns the rotated image and the new place of the nipple.
"""


def rotate(image, angle, top_point, nipple_point):
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
    print("(h, w) After rotate = ", (h, w))
    nipple_point = find_new_dot(nipple_point[0], nipple_point[1], angle, (cX, cY))
    x_rotated_nipple, y_rotated_nipple = nipple_point
    # cv2.circle(rotated, (int(x_rotated_nipple), int(y_rotated_nipple)), radius=30, color=(255, 0, 0), thickness=20)
    cv2.imwrite("image_after_rotate.png", rotated)

    # Stage 3 - shift the image to the left:
    num_rows, num_cols = rotated.shape[:2]
    translation_matrix = np.float32([[1, 0, -x_new], [0, 1, 0]])
    img_translation = cv2.warpAffine(rotated, translation_matrix, ((num_cols, num_rows)))
    (h, w) = img_translation.shape[:2]
    x_shifted_nipple, y_shifted_nipple = x_rotated_nipple - x_new, y_rotated_nipple
    nipple_point = (x_shifted_nipple, y_shifted_nipple)
    cv2.imwrite("image_after_shift.png", img_translation)
    print("(h, w) After shift = ", (h, w))
    return img_translation, nipple_point


"""
4. Compute polygon
"""


# TODO: (Priority = 2) More accurate
def ransac_polyfit(countors, center_point, h, w, source, isRotated):
    x_nipple, _ = center_point

    """ --- part of another method we tried that includes bounding boxes --- """
    # width_box = x_nipple * 1 / 2  # half of the width .
    # bbox = BoundingBox.from_center(center_point, width=w, height=2500)  # bounding box from center point
    # bbox = BoundingBox([(), ()])  # bounding box from center point
    # temp_image = np.copy(source)
    # cv2.rectangle(temp_image, (1497 - 500, 3727 - 1250), (1497 + 500, 3727 + 1250), (255, 0, 0), 20)

    cx, cy = center_point
    # unzip to x and y arrays:
    bottom_arr_x = np.array([])
    bottom_arr_y = np.array([])
    top_arr_x = np.array([])
    top_arr_y = np.array([])
    poly_top = []
    poly_bottom = []

    # Sort all contours on order to find biggest contour area that should include the breast contours.
    countors = sorted(countors, key=cv2.contourArea, reverse=True)[:1]
    original_image = np.copy(source)
    for c in countors:
        cv2.drawContours(original_image, [c], -1, (255, 0, 0), 10)
        '''
        cv2.namedWindow('Contours_By_Area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours_By_Area', 600, 600)
        cv2.imshow('Contours_By_Area', original_image)
        cv2.waitKey(0)
        '''
    ### draw the biggest contour that we gonna use (The frame of the image will be deleted later on)
    image_with_contours_1 = np.copy(source)
    image_with_contours = cv2.drawContours(image_with_contours_1, countors, -1, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imwrite("image_with_contours_1.png", image_with_contours)
    ###

    for a in countors:
        for b in a:
            # optional to add - blacken noise from the nipple to the right.
            if b[0][0] > cx + 50:
                continue
            # if the contour on the border of the image - ignore it.
            if b[0][1] <= 5 or b[0][1] >= h - 5 or b[0][0] >= w - 5:
                continue

            """ --- part of another method we tried that includes bounding boxes --- """
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

    top_muscle = gradient_compare(poly_top, center_point, "Top", isRotated)
    bottom_muscle = gradient_compare(poly_bottom, center_point, "Bottom", isRotated)
    coeff_top = RANSAC.quadratic_ransac_curve_fit("Upper Polynomial", top_arr_x, top_arr_y)
    coeff_bottom = RANSAC.quadratic_ransac_curve_fit("Lower Polynomial", bottom_arr_x, bottom_arr_y)
    return (coeff_top, coeff_bottom, top_muscle, bottom_muscle)


def gradient_compare(poly, nipple_point, flag, isRotated):
    xy = (-1, -1)
    x_old = nipple_point[0]
    y_old = nipple_point[1]
    last_ten = []
    res = (-1, -1)
    list_of_res = []
    res_counter = dict()
    avg_ten_m = -100
    old_ten_avg = -100
    n = 20
    if flag == "Bottom":
        counter_to_ten = 0
        m = (poly[0][1] - nipple_point[1]) / (poly[0][0] - nipple_point[0])
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
            # Case for first n contours to fill the list
            if counter_to_ten < n - 1:
                last_ten.append([m, (x, y)])
                counter_to_ten += 1
            # After filling n places in queue we starts to calculate avg of the gradient for each n contours
            elif counter_to_ten == n - 1:
                # Cases like this are probably noises
                if y_old - y > 40:
                    continue
                last_ten.append([m, (x, y)])
                only_m = []

                # Create list of n current gradients
                for j in range(0, len(last_ten)):
                    only_m.append(last_ten[j][0])

                # Calculate avg, find maximum gradient and create array of counters for each contour that has been choosen to be the biggest between all n.
                avg_ten_m = sum(only_m) / n
                index_max_m = np.argmax(only_m)
                res = last_ten[index_max_m][1]
                list_of_res.append(res)
                res_counter = Counter(list_of_res)

                # Calculate maximum y for all n current contours
                y_max = sorted(list_of_res, key=lambda x: x[1], reverse=True)
                print(" y_max = ", y_max)
                if -1 < avg_ten_m < 0 and x_old - x <= 20 and avg_ten_m > old_ten_avg and y_old < y and abs(
                        y_old - y) < 100:
                    if not first_time:
                        list_of_res.append(last_ten[-1][1])
                        first_time = True
                    print("Point found!!")
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[0][0])
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", Counter(res_counter).most_common())
                else:
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[-1][0])
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" y_max = ", y_max)
                    print(" Max res = ", Counter(res_counter).most_common())
                last_ten.pop(0)
                old_ten_avg = avg_ten_m

            x_old = x
            y_old = y
            m_old = m
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
                if isRotated == "yes":
                    index_m = np.argmin(only_m)
                else:
                    index_m = np.argmax(only_m)
                res = last_ten[index_m][1]
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
                    print(" Max res = ", Counter(res_counter).most_common())
                    # break
                else:
                    print(" x ,y  = ", (x, y))
                    print(" current point m = ", last_ten[-1][0])
                    print(" tetha_curr = ", tetha_curr)
                    print(" avg_curr = ", avg_ten_m)
                    print(" x ,y res = ", res)
                    print(" Max res = ", Counter(res_counter).most_common())
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

"""
3. run on all the images: 
Run on all the images, calculate length and width and store the results in two arrays.
After that, calculate for each array its AVG.
"""


def sum_calc(size_breast_list):
    widths = 0
    lengths = 0

    for i in size_breast_list:
        _, _, length, width = i
        widths += width
        lengths += length

    num_of_pics = len(size_breast_list)
    return widths, lengths, num_of_pics


"""
3. ratio between the width and length avg
"""


def ratio_calc(list_of_wh):
    widths, lengths, num_of_pics = list_of_wh
    '''
    widths = 0
    lengths = 0
    sum = 0

    for i in list_of_wh:
        w, h, size = i
        widths += w
        lengths += h
        num_of_pics += size
    '''
    ratio = (widths / num_of_pics) / (lengths / num_of_pics)
    return ratio


# crop from the image only the brest .
# y top - is the top of the picture , so its number smaller then the y_bottom because of the coordinates.
def crop_breast_from_image(x_nipple, y_top, y_bottom, image):
    cropped_image = image[y_top:y_bottom, 0:x_nipple + 100]
    return cropped_image


# calculates the change of image size by ratio info.
# the ratio calculated within the function "ratio_calc"
def change_image_by_ratio(xy_breast_list, list_after_ration, ratio_avg):
    for i in xy_breast_list:
        _,_,h, w = i
        add_for_x = ratio_avg * h - w
        w = w + add_for_x
        list_after_ration.append((h, w))
    return list_after_ration


# checks the max x and max y in list.
def max_image_size(list_after_ration, x_max=0, y_max=0):
    for i in list_after_ration:
        y, x = i
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    return y_max,x_max


# sizes of max image - x_max,y_max
def change_image_size(image, path, index, list_of_sizes, x_max, y_max):
    # First resize image by ratio
    h, w = list_of_sizes[index]
    dim = (int(w), int(h))

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Second paste image on blank image with max size
    print(int(y_max))
    print(int(x_max))
    blank_image = np.zeros((int(y_max), int(x_max), 3), np.uint8)
    blank_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image

    cv2.imwrite(path + "\\" + "new.png", blank_image)


def find_new_dot(x, y, angle, center):
    angle_rad = math.radians(angle)
    cX, cY = center
    print("angle_rad = ", angle_rad)
    x_new = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_new = -(x - cX) * np.sin(angle_rad) + (y - cY) * np.cos(angle_rad) + cY
    return (x_new, y_new)


def run_processing(source, tagged,path):
    h,w = tagged.shape[:2]
    print(w,h)
    isRotated = "no"
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

    if angle > 90:
        source = remove_white_frame_norotate(source, top_point, buttom_point)

    if angle < 90:
        isRotated = "yes"
        # Delete white frame
        source = remove_white_frame(source, top_point)

        source_rotated, nipple_point = rotate(source, angle, top_point, nipple_point)
        # roi_rotated, nipple_point = rotate(roi, angle, top_point, nipple_point)
        source = source_rotated

    # RANSAC:
    # TODO: (Priority = 1) More accurate contours, with minimum noise + Check on several images - Done!.
    countors = find_countor.getEdgeImage(source)
    poly_top, poly_bottom, top_muscle, bottom_muscle = ransac_polyfit(countors, nipple_point, h, w, source, isRotated)
    x1, y_bottom = bottom_muscle
    x2, y_top = top_muscle

    # TODO: (Priority = 3) Check if true, after get to the best result.
    # calculate new length of muscle:
    length_breast = Finding_Length(top_muscle, bottom_muscle)

    # calculate new width of muscle:
    width_breast = nipple_point[0]

    # get new shape of image after pre-processing:
    h_image, w_image = source.shape[:2]
    cropped_image = crop_breast_from_image(int(width_breast), int(y_top), int(y_bottom), source)
    cv2.imwrite(path+"\\"+"source.png",cropped_image)
    # print(" info image: (y_bottom, y_top, length_breast, width_breast) = ", (y_bottom, y_top, length_breast, width_breast))
    return source, (y_bottom, y_top, length_breast, width_breast)
    # , roi_rotated add to return!!


"""
This function returns all png files in subs folders of root folder
"""


def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files