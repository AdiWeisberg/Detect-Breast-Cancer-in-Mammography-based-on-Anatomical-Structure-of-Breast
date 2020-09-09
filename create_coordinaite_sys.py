import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import find_borders, find_countor
import math
import imutils

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


# finding the normal of the muscle line for the nipple line.
# find the the slope of the line of the muscale(m1) and then calculate the normal:
# m2 = -1 / m1.
# return the normal number.
def Finding_Normal(line, point1=[1, 2]): # remember to change the point1 .
    x = np.array(point1)
    m, b = line
    m_normal = -1 / m
    print(m_normal)
    plt.plot(x, m_normal * x + b)
    plt.show()
    return m_normal


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


# we have all the point on the breast countor by the code in- find_countor and we will find the parabola by
# Three or more central points as far apart as possible.
# return parabola equation
# TODO BY NAOMI
def Finding_Prabola_By_Countor():
    pass


# by Intersection Point betweeen Parabola and LineMuscle
# return two point -first and last on the muscle.
# TODO BY: NAOMI
def Finding_Inter_Point_Between_Parabola_LineMuscle(Parabola, LineMuscle):
    pass


# TODO BY: NAOMI
# return the length of the full muscla line.
def Finding_Length(pointFirst, pointlast):
    pass


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


def rotate(image, angle, top_line):
    # Stage 1 - rotation the image by the angle.
    rotated = imutils.rotate_bound(image, -angle)  # think to be a better option then the ndimage.rotate

    # Stage 2 - Calculate Distance to shift:
    x, y = top_line
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

def ransac_polyfit(countors, order=3, n=20, k=100, t=0.1, d=100, f=0.8):
    # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus

    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    # unzip to x and y arrays:
    arr_x = np.array([])
    arr_y = np.array([])
    for a in countors:
        for b in a:
            # print(" b is : ", b[0])
            arr_x = np.append(arr_x, b[0][0])
            arr_y = np.append(arr_y, b[0][1])
    print(" arr_x : ", arr_x)
    print("len(arr_x) -> ", len(arr_x))
    print("len(arr_y) -> ", len(arr_y))

    besterr = np.inf
    bestfit = 0.0
    for kk in range(k):
        maybeinliers = np.random.randint(len(arr_x), size=n)
        maybemodel = np.polyfit(arr_x[maybeinliers], arr_y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, arr_x) - arr_y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(arr_x) * f:
            bettermodel = np.polyfit(arr_x[alsoinliers], arr_y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, arr_x[alsoinliers]) - arr_y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    poly = np.poly1d(bestfit)
    new_x = np.linspace(arr_x[0], arr_x[-1])
    new_y = poly(new_x)
    plt.plot(arr_x, new_y, "o", new_x, new_y)
    plt.xlim([arr_x[0] - 1, arr_x[-1] + 1])
    plt.savefig("line.jpg")

    return bestfit


def find_new_dot(x, y, angle):
    angle_rad = math.radians(angle)
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return (x_new, y_new)

def main():
    # Read an image
    image = cv2.imread("1-2.png")
    print("shapes: ", image.shape[:2])
    top_line, buttom_line = find_borders.findLine(image)
    output, circle, center_point = find_borders.findCircle(image)

    # draw and show line between 2 points that we found
    line_detected = np.copy(image)
    #cv2.line(line_detected, top_line, buttom_line, (208, 216, 75), 15)

    # Find Equation line and angle for rotation
    eq_line_muscle = Finding_Equation_Line(top_line, buttom_line)
    angle = angle_calc(eq_line_muscle)
    print("angle = ", angle)
    if angle < 90:
        rotated = rotate(image, angle, top_line)
        image = rotated

    # RANSAC:
    # image_copy = np.copy(image)
    # countors, edgeImage = find_countor.getEdgeImage(image_copy)
    # poly_breast = ransac_polyfit(countors)

    normal = Finding_Normal(eq_line_muscle)
    eq_line_width = Finding_Equation_Line_By_Slope_And_Point(normal, center_point)
    intercept_width_length = Find_intercept_width_length(eq_line_muscle, eq_line_width)

    # view images:
    cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output1', 600, 600)
    cv2.imshow("output1", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
