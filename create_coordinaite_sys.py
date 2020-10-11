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
        cv2.namedWindow('Contours_By_Area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours_By_Area', 600, 600)
        cv2.imshow('Contours_By_Area', original_image)
        cv2.waitKey(0)

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


def find_new_dot(x, y, angle, center):
    angle_rad = math.radians(angle)
    cX, cY = center
    print("angle_rad = ", angle_rad)
    x_new = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_new = -(x - cX) * np.sin(angle_rad) + (y - cY) * np.cos(angle_rad) + cY
    return (x_new, y_new)


def run_processing(source, tagged, roi):
    h, w = tagged.shape[:2]
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
    x1, y1 = bottom_muscle
    x2, y2 = top_muscle

    # bbox = BoundingBox.from_center(nipple_point, width=1000, height=2500)  # bounding box
    # cv2.rectangle(source, (1497-500 , 3727-1250), (1497+500, 3727+1250),(255, 0, 0),20)

    # TODO: (Priority = 3) Check if true, after get to the best result.
    # calculate new length of muscle:
    length_breast = Finding_Length(top_muscle, bottom_muscle)

    # calculate new width of muscle:
    width_breast = nipple_point[1]

    # get new shape of image after pre-processing:
    h_image, w_image = source.shape[:2]

    print(" info image: (h_image, w_image, length_breast, width_breast) = ", (h_image, w_image, length_breast, width_breast))
    return source, (h_image, w_image, length_breast, width_breast)
    # , roi_rotated add to return!!

"""
This function returns all png files in subs folders of root folder
"""

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

def main():
    # Read tagged and source images:

    # --- rotated muscle line images: ---
    # tagged = cv2.imread("images\\Mass-Test_P_01140_LEFT_MLO_tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_01140_LEFT_MLO1-1.png")
    # tagged = cv2.imread("images\\Mass-Test_P_01348_LEFT_MLO_tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_01348_LEFT_MLO.png")
    # tagged = cv2.imread("images\\Calc-Test_P_01883_RIGHT_MLO_tagged.png")
    # source = cv2.imread("images\\Calc-Test_P_01883_RIGHT_MLO.png")

    # --- straight muscle line images: ---
    # tagged = cv2.imread("images\\Mass-Test_P_01719_RIGHT_CC_tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_01719_RIGHT_CC.png")
    # tagged = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC_Tagged.png")
    # source = cv2.imread("images\\Mass-Test_P_00699_RIGHT_CC.png")

    result_path = "E:\\breast_dataset_test"
    os.makedirs(result_path + "\\" + "Train" + "\\" + "Mass")
    os.makedirs(result_path + "\\" + "Train" + "\\" + "Calc")
    os.makedirs(result_path + "\\" + "Test" + "\\" + "Mass")
    os.makedirs(result_path + "\\" + "Test" + "\\" + "Calc")
    os.makedirs(result_path + "\\" + "Val" + "\\" + "Calc")
    os.makedirs(result_path + "\\" + "Val" + "\\" + "Mass")
    test_mass = result_path + "\\" + "Test" + "\\" + "Mass"
    test_calc = result_path + "\\" + "Test" + "\\" + "Calc"

    # for split the train to 70,30 for the val:
    train_mass = result_path + "\\" + "Train" + "\\" + "Mass"
    train_calc = result_path + "\\" + "Train" + "\\" + "Calc"
    val_mass = result_path + "\\" + "Val" + "\\" + "Mass"
    val_calc = result_path + "\\" + "Val" + "\\" + "Calc"

    start_path = "E:\\TCIABreast"
    calc_train_path = start_path + "\\Calc-Training"
    mass_train_path = start_path + "\\Mass-Training\\CBIS-DDSM"
    mass_test_path = start_path + "\\Mass-Test"
    calc_test_path = start_path + "\\Calc-Test\\CBIS-DDSM"

    path_list = [("Calc-Training", calc_train_path, "Train", "Calc", train_calc),
                 ("Mass-Training", mass_train_path, "Train", "Mass", train_mass),
                 ("Mass-Test", mass_test_path, "Test", "Mass", test_mass),
                 ("Calc-Test", calc_test_path, "Test", "Calc", test_calc)]

    # iterate each folder and image and get the image processing done
    list_of_folders = []
    for folder_name, path, flag1, flag2, save_path in path_list:
        all_images_specific_folder = os.listdir(path)
        print("path_images = ", all_images_specific_folder)
        folder_lst = []
        # iterate over all the images in specific folder
        for n, image_name in enumerate(sorted(all_images_specific_folder)):
            print("folder_name, image_name = ", folder_name, image_name)
            sub1 = os.listdir(path + "\\" + image_name)
            sub2 = os.listdir(path + "\\" + image_name + "\\" + sub1[0])
            full_path = os.path.join(path, image_name, sub1[0], sub2[0])
            files_in_folder = os.listdir(full_path)
            """ if 1 picture is wrong the program will stop and we should run again after fix the problem from the place we stopped! """
            if (len(files_in_folder) != 2):
                raise Exception('there is not png image in directory: {} '.format(full_path))
                exit(0)

            # Specify the output folder path that will contain 2 files - source and tagged image.
            new_location = result_path + "\\" + flag1 + "\\" + flag2
            # os.makedirs(new_location)

            # convert dcm to png and copy it to new path:
            dcm_to_png_file = full_path + "\\1-1.dcm"
            ds = pydicom.dcmread(dcm_to_png_file)
            pixel_array_numpy = ds.pixel_array
            # TODO: Michal use this for the second big for to save iamges
            #save_path_source = new_location + "\\" + str(n) + ".png"
            #print("dcm to png for file " + str(n) + " succeeded? ",
            #      str(cv2.imwrite(save_path_source, pixel_array_numpy)))

            # Read images: source and tagged from new folders
            source = cv2.imread(files_in_folder[0])
            if files_in_folder[0].endswith(".png"):
                tagged = cv2.imread(files_in_folder[0])
            else:
                tagged = cv2.imread(files_in_folder[1])
            # TODO: Naomi - check if 2 file or 1 is needed!
            # roi = cv2.imread(files_in_folder[??])
            roi - []  # delete after set imread for roi
            shifted_image, image_info = run_processing(source, tagged, roi)
            # This code copy the tagged image to new path - but we don't need it after the processing.
            # png_new_location = new_location + "\\" + n + ".png"
            # png_old_location = full_path + "\\" + "1-1.png"
            # shutil.copy(png_old_location, png_new_location)

            # Save the new shifted image at the same old path
            cv2.imwrite(full_path + "\\" + str(n) + ".png", shifted_image)

            lst.append(image_info)
        list_of_folders.append(folder_lst)

    # TODO: Michal - take this list and work with it. the updated images will be orginized at the tree folder that we talked about under folder of the name of the image
    #  for each shifted image. Each image will be saved as
    print(list_of_folders)

    # split the data from train to val:
    mass_train_list = get_file_list_from_dir(train_mass)  # list of all the images of mass train
    calc_train_list = get_file_list_from_dir(train_calc)  # list of all the images of calc train

    print(" mass_train_list = ", mass_train_list)
    print(" calc_train_list = ", calc_train_list)

    len_mass = floor(mass_train_list.__len__() * 0.3)
    len_calc = floor(calc_train_list.__len__() * 0.3)

    i = 0
    for im in mass_train_list:
        if i <= len_mass:
            i = i + 1
            print(train_mass + "\\" + im)
            shutil.copy(train_mass + "\\" + im, val_mass + "\\" + im)
    i = 0
    for im in calc_train_list:
        if i <= len_calc:
            i = i + 1
            shutil.copy(train_calc + "\\" + im, val_calc + "\\" + im)


if __name__ == '__main__':
    main()
