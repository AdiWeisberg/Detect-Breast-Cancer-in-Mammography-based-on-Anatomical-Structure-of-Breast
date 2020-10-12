from utils import *
from image_processor import change_image_by_ratio, max_image_size

def main():
    # Read tagged and source images:

    global flag
    start_path = "D:/images"

    folder_lst = image_size_locator(start_path)
    sum_lst = calculate_whole_image_size(folder_lst)
    ratio_num = calculate_image_ratio(sum_lst)
    lst_new = []
    list_new = change_image_by_ratio(folder_lst, lst_new, ratio_num)

    y_max, x_max = max_image_size(list_new)
    image_resize_by_ratio(start_path, list_new, x_max, y_max)


if __name__ == '__main__':
    main()