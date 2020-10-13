import cv2
import os
from image_processor import run_processing
from image_processor import change_image_size, ratio_calc, sum_calc
from image_format import *

def image_size_locator(start_path):
    folder_lst = []
    path_list = os.listdir(start_path)

    for calc_mass_folders in path_list:
        mid_path = os.path.join(start_path, calc_mass_folders)
        images_folders = os.listdir(mid_path)
        for num_folder in images_folders:
            full_path = os.path.join(start_path, calc_mass_folders, num_folder)
            source_path = full_path + "\\" + "source.png"

            dcm_to_png(full_path)
            source = cv2.imread(source_path)
               
            # Flip if the source is not at the right direction
            if is_flip_to_left(source):
                source = cv2.flip(source, 1)  # flip the image that the brest will be in the left side
            tagged = cv2.imread(full_path + "\\" + "1-1.png")

            shifted_image, image_info = run_processing(source, tagged, full_path)
            folder_lst.append(image_info)
    return folder_lst


def image_resize_by_ratio(start_path, list_new, x_max, y_max):
    path_list = os.listdir(start_path)
    for calc_mass_folders in path_list:
        mid_path = os.path.join(start_path, calc_mass_folders)
        images_folders = os.listdir(mid_path)
        for num_folder in images_folders:
            index = 0
            full_path = os.path.join(start_path, calc_mass_folders, num_folder)
            source = cv2.imread(full_path + "\\" + "source.png")
            change_image_size(source, full_path, index, list_new, x_max, y_max)
            index = index + 1


def calculate_image_ratio(sum_lst):
    return ratio_calc(sum_lst)


def calculate_whole_image_size(image_info_lst):
    return sum_calc(image_info_lst)

