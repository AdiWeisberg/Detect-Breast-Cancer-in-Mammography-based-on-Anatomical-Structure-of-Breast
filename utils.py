import cv2
import os
from image_processor import run_processing
from image_processor import change_image_size, ratio_calc, sum_calc
from image_format import *
from image_processor import draw_roi

def image_size_locator(start_path):
    folder_lst = []
    path_list = os.listdir(start_path)

    for calc_mass_folders in path_list:
        mid_path = os.path.join(start_path, calc_mass_folders)
        images_folders = os.listdir(mid_path)
        for num_folder in images_folders:
            full_path = os.path.join(start_path, calc_mass_folders, num_folder)
            mask = cv2.imread(full_path + "\\" + "mask.png", cv2.IMREAD_GRAYSCALE)

            # Convert dcm to png:
            dcm_to_png(full_path)
            source = cv2.imread(full_path + "\\" + "source.png")

            # flip the image and the mask so that the brest will be in the left side if needed
            if is_flip_to_left(source):
                source = cv2.flip(source, 1)
                mask = cv2.flip(source, 1)
                cv2.imwrite(full_path + "\\" + "source.png", source)
            tagged = cv2.imread(full_path + "\\" + "1-1.png")

            # Run image processing, shifted_image is the result and shifted_info is the measurements of the breast and new image
            shifted_image, image_info = run_processing(source, tagged, full_path, mask)

            folder_lst.append(image_info)
    return folder_lst


def calculate_image_ratio(sum_lst):
    return ratio_calc(sum_lst)


def image_resize_by_ratio(start_path, list_new, x_max, y_max):
    path_list = os.listdir(start_path)
    for calc_mass_folders in path_list:
        mid_path = os.path.join(start_path, calc_mass_folders)
        images_folders = os.listdir(mid_path)
        for index, num_folder in enumerate(images_folders):
            full_path = os.path.join(start_path, calc_mass_folders, num_folder)
            source = cv2.imread(full_path + "\\" + "source.png")
            mask = cv2.imread(full_path + "\\" + "mask_cropped.png")
            change_image_size("final_ratio.png", source, full_path, index, list_new, x_max, y_max)
            change_image_size("final_mask.png", mask, full_path, index, list_new, x_max, y_max)
            draw_roi(full_path, "final_ratio.png", "final_mask.png", "final_roi.png")

def calculate_whole_image_size(image_info_lst):
    return sum_calc(image_info_lst)

