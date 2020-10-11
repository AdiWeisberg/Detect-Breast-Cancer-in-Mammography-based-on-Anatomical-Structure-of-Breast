import cv2
import os
from image_procesor import run_processing
from image_procesor import change_image_size, ratio_calc, sum_calc


def image_size_locator(path_list):
    folder_lst = []

    for folder_name, path in path_list:
        all_images_specific_folder = os.listdir(path)

        source = cv2.imread(path + "\\" + all_images_specific_folder[0])
        tagged = cv2.imread(path + "\\" + all_images_specific_folder[1])

        shifted_image, image_info = run_processing(source, tagged, path)
        folder_lst.append(image_info)
    return folder_lst


def image_resize_by_ratio(path_list, list_new, x_max, y_max):
    for folder_name, path in path_list:
        index = 0
        all_images_specific_folder = os.listdir(path)

        source = cv2.imread(path + "\\" + all_images_specific_folder[0])
        change_image_size(source, path, index, list_new, x_max, y_max)
        index = index + 1


def calculate_image_ratio(sum_lst):
    return ratio_calc(sum_lst)


def calculate_whole_image_size(image_info_lst):
    return sum_calc(image_info_lst)
