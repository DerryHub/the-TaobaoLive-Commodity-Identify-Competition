import time
import numpy as np


def get_time_str():
    return time.strftime("%Y-%m-%d, %H:%M:%S ", time.localtime((time.time()) ))

def print_info(msg):
    print(get_time_str(), msg) 

# saving data into npy
def data_to_pkl(data, file_path):
    print("Saving data to file(%s). "%(file_path))
    np.save(file_path, data)

def save_draw_file(draw_list):
    st = ""
    for row in draw_list:
        threshold = row[1]
        st += str(threshold) + " "
        for r in row[0]:
            for c in r:
                st += str(c) + " "
        st += "\n"
    with open("draw_file.txt", "w") as f:
        f.write(st)


def report_format(report):
    report = report.split()
    result = np.zeros((3,3))
    result[0][0] = report[5]
    result[0][1] = report[6]
    result[0][2] = report[7]

    result[1][0] = report[10]
    result[1][1] = report[11]
    result[1][2] = report[12]

    result[2][0] = report[17]
    result[2][1] = report[18]
    result[2][2] = report[19]

    return result