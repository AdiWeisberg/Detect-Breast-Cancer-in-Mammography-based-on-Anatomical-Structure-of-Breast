from utils import *
from image_procesor import change_image_by_ratio, max_image_size


def main():
    # Read tagged and source images:

    global flag
    start_path = "C:\py_projects\Detect-Breast-Cancer-in-Mammography-based-on-Anatomical-Structure-of-Breast\images"
    pic1 = start_path + "\\1"
    pic2 = start_path + "\\2"

    path_list = [("pic1", pic1),
                 ("pic2", pic2)]

    folder_lst = image_size_locator(path_list)
    sum_lst = calculate_whole_image_size(folder_lst)
    ratio_num = calculate_image_ratio(sum_lst)
    lst_new = []
    list_new = change_image_by_ratio(folder_lst, lst_new, ratio_num)

    y_max, x_max = max_image_size(list_new)
    image_resize_by_ratio(path_list, list_new, x_max, y_max)


if __name__ == '__main__':
    main()
