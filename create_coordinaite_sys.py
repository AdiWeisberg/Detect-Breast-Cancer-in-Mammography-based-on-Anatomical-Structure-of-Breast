import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import find_borders, find_countor, RANSAC
import math
import imutils


"""
0. Delete white frame:
"""

def remove_white_frame(image, top_point):
    h, w = image.shape[:2]
    # from bottom:
    for i in range(h-150, h):
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
1. finding width of the coordinate:
"""


# FINDING THE LINE OF MUSCLE!

# Find an equation straight to the muscle line you tagged
# by the first and the last point we found in find_borders
# return line object
def Finding_Equation_Line(point1, point2):
    point1 = np.array(point1)
    print(point1)
    point2 = np.array(point2)
    print(point2)
    z = np.polyfit(point1, point2, 1)
    m, b = z
    print(m, b)
    line_length = np.poly1d(z)
    plt.plot(point1, m * point1 + b)
    plt.show()
    return line_length

# for the nipple line:
# circle - get the circle ( the circle taf of the nipple )
# slope - the number we get from the function "finding normal" that calculates the slope
# y-y1 = m2(x-x1)
# return line
def Finding_Equation_Line_By_Slope_And_Point(slope, center_point):
    x, y, r = center_point
    m = slope
    b = y - slope * x
    line_width = np.poly1d((m, b))
    return line_width


# by Intersection Point between line nipple and line muscle.
# return width number
# x = (b2 - b1) / (m1 - m2)
# put x in some equation and got y
def Find_intercept_width_length(line_length, line_width):
    m1, b1 = line_length
    m2, b2 = line_width
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)


# This function gets 2 points of the nipple and the iterception point with the
# length and width equation lines.
# the function returns the width in float.
def Find_Width(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Calculating distance
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


"""
2. finding Length
"""

# TODO BY: NAOMI
# return the length of the full muscla line.
def Finding_Length(poly_breast):
    poly_top, poly_bottom = poly_breast
    return (poly_bottom - poly_top)

"""
3. run on all the images:
big for that run on all the images and calculate the length and the width and store them in  two arrays.
and then calculate for each array its AVG.
"""

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


def rotate(image, angle, top_point):
    # Stage 1 - rotation the image by the angle.
    rotated = imutils.rotate_bound(image, -angle)  # think to be a better option then the ndimage.rotate

    # Stage 2 - Calculate Distance to shift:
    x, y = top_point
    new_point = find_new_dot(x, y, angle)
    x_new, y_new = new_point
    cv2.circle(rotated, (int(x_new), int(y_new)), radius=10, color=(0, 0, 255), thickness=10)

    # Stage 3 - shift the image to the left:
    num_rows, num_cols = rotated.shape[:2]
    translation_matrix = np.float32([[1, 0, -x_new], [0, 1, 0]])
    img_translation = cv2.warpAffine(rotated, translation_matrix, ((num_cols, num_rows)))
    return img_translation



"""
4. Compute polygon
"""

def ransac_polyfit(countors, center_point):
    cx, cy, r = center_point
    # unzip to x and y arrays:
    bottom_arr_x = np.array([])
    bottom_arr_y = np.array([])
    top_arr_x = np.array([])
    top_arr_y = np.array([])
    for a in countors:
        for b in a:
            if b[0][1] <= cy:
                top_arr_x = np.append(top_arr_x, b[0][0])
                top_arr_y = np.append(top_arr_y, b[0][1])
            else:
                bottom_arr_x = np.append(bottom_arr_x, b[0][0])
                bottom_arr_y = np.append(bottom_arr_y, b[0][1])

    print(" top_arr_x ", len(top_arr_x))
    print(" top_arr_y ", len(top_arr_y))

    print("TOP1")
    coeff_top = RANSAC.quadratic_ransac_curve_fit(top_arr_x, top_arr_y)
    print("BOTTOM1")
    coeff_bottom = RANSAC.quadratic_ransac_curve_fit(bottom_arr_x, bottom_arr_y)

    return coeff_top, coeff_bottom



def find_new_dot(x, y, angle):
    angle_rad = math.radians(angle)
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return (x_new, y_new)

def main():
    # Read an image
    image = cv2.imread("1-7.png")
    print("shapes: ", image.shape[:2])

    # Find line of muscle and nipple
    top_point, buttom_point = find_borders.findLine(image)

    # Delete white frame
    image = remove_white_frame(image, top_point)

    # draw and show line between 2 points that we found
    #line_detected = np.copy(image)
    #cv2.line(line_detected, top_point, buttom_point, (208, 216, 75), 15)

    # Find Equation line and angle for rotation
    eq_line_muscle = Finding_Equation_Line(top_point, buttom_point)
    angle = angle_calc(eq_line_muscle)
    print("angle = ", angle)
    if angle < 90:
        rotated = rotate(image, angle, top_point)
        image = rotated

    # Find nipple
    circle, nipple_point = find_borders.findCircle(image)
    x_new, y_new, r = nipple_point
    cv2.circle(image, (int(x_new), int(y_new)), radius=30, color=(0, 0, 255), thickness=20)

    # view images:
    cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output1', 600, 600)
    cv2.imshow("output1", image)
    cv2.waitKey(0)

    # RANSAC:
    countors = find_countor.getEdgeImage(image)
    poly_breast = ransac_polyfit(countors, nipple_point)
    print(type(poly_breast[0]))
    # calculate new length of muscle:
    muscle_length = Finding_Length(poly_breast)

    # calculate width:
    width_length = nipple_point[0]
    width_equation = np.array(nipple_point[1])
    width_iter_muscle = (0, nipple_point[1]) # The intercetion point between the muscle line to the width line.


if __name__ == '__main__':
    main()