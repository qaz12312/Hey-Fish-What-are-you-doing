"""
(測試環境用)將 Deeplabcut 的 csv 檔中的資料轉成 LSTM 所需格式.
"""
from csv import reader
import numpy as np
import copy
from os import getenv, listdir, makedirs
from os.path import isfile, join, isdir
from dotenv import load_dotenv
import argparse
import sys
from traceback import extract_tb
import FishLog

load_dotenv()
parser = argparse.ArgumentParser(description="Run convert data to LSTM.")
parser.add_argument('--steps', help='n_steps',type=int, default=30)
parser.add_argument('--jumps', help='n_jumps',type=int, default=2)
parser.add_argument('--normal', help='is_normal',type=int, default=0)
parser.add_argument('--len', help='n_len',type=int, default=0)
parser.add_argument('--translation', help='is_translation',type=int, default=0)
parser.add_argument('--rotate', help='is_rotate',type=int, default=0)
args = parser.parse_args()

PROJECT_PATH = getenv('PROJECT_PATH')
NOT_USED_LIST = [int(x) for x in getenv('NOT_USED').split(',')]
'''
不使用的的座標點
'''
CHECK_FRONT_BACK = [int(x) for x in getenv('CHECK_FRONT_BACK').split(',')]
'''
frame 為正背面時, 不會有的座標點
'''
TOTAL_POINTS = int(getenv('TOTAL_POINTS')) - len(NOT_USED_LIST)
'''
1 個 frame 有 n 個座標點
'''
N_STEPS = args.steps
'''
n frame 為一個 action
'''
temp_jump = args.jumps
if temp_jump < 1:
    temp_jump = 1
JUMP_N_FRAME = temp_jump
'''
一秒取 30/n 個 frame
'''
# for 正規化
IS_NORMAL = args.normal
# scale
SCALE_LEN = args.len
'''
MIRROR_POINT 到 TRANSLATION_POINT 的長度 = n (0:不縮放)
'''
# translation
IS_TRANSLATION = args.translation
'''
是否要做平移 (type:0/1)
'''
TRANSLATION_POINT = int(getenv('TRANSLATION_POINT'))
'''
以第 n 個座標點為基準點(0,0)
'''
# mirror
IS_ROTATE = args.rotate
'''
正規化資料時, 是否檢查旋轉(2:就檢查鏡像) (type: 0/1/2)
'''
MIRROR_POINT = int(getenv('MIRROR_POINT'))
'''
以第 n 個座標點是否為負來判斷是否鏡像(旋轉) (type: 0/1)
'''
# for current file
_current_file_path = ''
'''
目前正在執行的csv檔的絕對路徑
'''
# -----------------------------------------------------------
# Read Deeplabcut log file (.csv) to get the coordinates that LSTM needs to use.
# -----------------------------------------------------------
def getCoords(filePath):
    """
    Read Deeplabcut log file (.csv) to get the coordinates that LSTM needs to use.

    Parameters
    ----------
    filePath : `str`
        The absolute path of the log file.

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D, 每一 dim 的元素個數 = [N(frame數), 3(x,y,z座標), 21-x(座標點)].
    """
    data_x = []

    with open(filePath, newline='') as f:
        frame_iter = reader(f)  # 讀取 CSV 檔案內容
        line = 0
        for frame_coords in frame_iter:
            line += 1
            if line < 4:  # 略過前3行
                continue
            # 第一個(coords)不需要
            temp_arr = np.array(frame_coords[1:], dtype=object)

            # 若有座標點 = '', 將 '' 轉成 None
            empty_indices_arr = (np.where(temp_arr == ''))[0]
            if empty_indices_arr.size > 0:
                temp_arr[empty_indices_arr] = np.newaxis

            # 轉成 float64 ndarray
            temp_arr = temp_arr.astype(np.float64)

            # 移除x,y的後面一個(likelihood)
            data_x.append([
                np.delete(temp_arr[::3], NOT_USED_LIST),   # x 座標
                np.delete(temp_arr[1::3], NOT_USED_LIST),  # y 座標
                np.ones(TOTAL_POINTS)                      # z 座標
            ])
    return np.array(data_x, dtype=object)
    """Return value
    [
        ...,
        [
            [frame n x],
            [frame n y],
            [frame n z]
        ],
        ...,
    ]
    """


# -----------------------------------------------------------
# Write data(3D coordinate array) into file (.txt)
# -----------------------------------------------------------
def writeData(dirPath, fileName, data_arr, n_actions, label=0):
    """
    Write data(3D coordinate array) into file (.txt).

    Parameters
    ----------
    dirPath : `str`
        The absolute path of the file directory.
    fileName : `str`
        e.g. train, validate, test
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(y*N_STEPS, 3, 21-x)
    n_actions : `int`
        y
    label: `int`, optional
        bigger than 0

    Returns
    -------
    `True`
    """
    with open(dirPath+'/X_'+fileName +'.txt', 'a') as f:
        for frame in data_arr:
            # frame_arr = [x1, y1, ..., xn, yn]
            frame_arr = frame[0][:]  # frame_arr = x 座標 array
            for idx in range(len(frame[1])):  # 將 y 座標 array insert 到 frame_arr
                frame_arr = np.insert(frame_arr, idx*2+1, frame[1][idx])
            f.write("{}\n".format(",".join(x for x in ["%s" % num for num in frame_arr])))
    if label > 0:
        with open(dirPath+'/Y_'+fileName+'.txt', 'a') as f:
            f.write("{}\n".format(label) * n_actions)
    return True


# -----------------------------------------------------------
# 填補空資料 & 將 frame 切成一個個 action & 將每個 action 的 frame 正規化
# -----------------------------------------------------------
def convertToUseful(data_arr):
    """
    註記: frame 是魚的正背面 (將其作為分段點).
    填補空座標點: 該 frame 是上下左右側 -> 將另一側的座標點複製過來
        其餘空值, 用預期值填上.
    將 frame 切成一個個 action 並正規化

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21-x)

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21-x)
    `int`
        number of actions
    """
    cut_list = []            # 要註記的 frame_number (從 0 開始算)
    interpolation_dict = {}  # { frame_number : [要填補的座標點] }

    for frame_idx in range(len(data_arr)):
        empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]  # 用 frame 的 x 座標找出值 = nan 的座標

        # 若 frame 是正背面 - 正: [背鰭(後), 尾巴(前偏上), 尾巴(前偏下)] / 背: [嘴, 背鰭(前), 腹鰭(前), 比腹鰭再前面的一點]
        is_front_back_arr = np.in1d(CHECK_FRONT_BACK, empty_indices_arr, assume_unique=True)
        if (np.sum(is_front_back_arr[:3]) == 3) or (np.sum(is_front_back_arr[3:]) == 4):
            cut_list.append(frame_idx)
        else:
            # 若 frame 是左右側 - 左: [右眼] / 右: [左眼]
            is_side_arr = np.in1d([0, 1], empty_indices_arr)
            for coord_idx in range(2):
                if is_side_arr[coord_idx]:
                    copy_idx = 0 if coord_idx == 1 else 1
                    data_arr[frame_idx][0][coord_idx] = data_arr[frame_idx][0][copy_idx]
                    data_arr[frame_idx][1][coord_idx] = data_arr[frame_idx][1][copy_idx]

            # 若仍有座標 = nan
            empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]
            if empty_indices_arr.size > 0:
                if frame_idx > 1:  # 因期望值是2次多項式
                    interpolation_dict[frame_idx] = empty_indices_arr
                else:
                    cut_list.append(frame_idx)

    if bool(interpolation_dict):  # if interpolation_dict is not empty
        print("需要進行填補的座標點 : \n{}".format(interpolation_dict))
        FishLog.writeLog(FishLog.formatLog(30, "TESTING_convertData.py", "line 224", "List the empty croods.", "[{}] empty croods: {}".format(_current_file_path, interpolation_dict)))
        data_arr = _fillInCoords(data_arr, interpolation_dict)

    return _frameSplit(data_arr, cut_list)


# -----------------------------------------------------------
# 將座標點補齊 - 用期望值填值
# -----------------------------------------------------------
def _fillInCoords(data_arr, interpolation_dict):
    """
    將座標點補齊 - 使用期望值填值.

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21-x)
    interpolation_dict : `dict[int, 1D int ndarray]`
        有缺的座標點 & 其 frame number (`{ frameNum: [coord_n,...], ...}`)

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21-x)
    """
    total_frames = len(data_arr)
    for frame_k, coords in interpolation_dict.items():
        for index in coords:
            value_py = []
            value_px = []
            frame_p = []

            up = frame_k + 1
            low = frame_k - 1
            while len(frame_p) < 3:
                if up < total_frames:
                    if data_arr[up, 0, index] == data_arr[up, 0, index]:
                        value_px.append(data_arr[up, 0, index])
                        value_py.append(data_arr[up, 1, index])

                        frame_p.append(up)
                    up += 1
                if low >= 0:
                    if data_arr[low, 0, index] == data_arr[low, 0, index]:
                        value_px.append(data_arr[low, 0, index])
                        value_py.append(data_arr[low, 1, index])

                        frame_p.append(low)
                    low -= 1

            fx = np.poly1d(np.polyfit(frame_p, value_px, 2))
            fy = np.poly1d(np.polyfit(frame_p, value_py, 2))
            data_arr[frame_k, 0, index] = fx(frame_k)
            data_arr[frame_k, 1, index] = fy(frame_k)

    return data_arr


# -----------------------------------------------------------
# 將 frame 切成一個個 action & 將每個 action 的 frame 正規化
# -----------------------------------------------------------
def _frameSplit(data_arr, cut_list):
    '''
    將 frame 切成一個個 action & 將每個 action 的 frame 正規化

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21-x)
    cut_list : `list`
        要註記的frame index

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(y*N_STEPS, 3, 21-x)
    `int`
        number of actions
    '''
    result_list = []
    cut_set = set(cut_list)
    total_actions = 0

    max_action_count = len(data_arr)-((N_STEPS-1)*JUMP_N_FRAME+1)+1  # 最多會有 y 個 actions 產生
    if max_action_count < 1 :
        print("資料量太少 Failed")
        return np.array([], dtype=object), total_actions
    for start_frame_idx in range(max_action_count):
        frame_in_action_range = range(start_frame_idx, start_frame_idx+((N_STEPS-1)*JUMP_N_FRAME+1), JUMP_N_FRAME)
        if set(frame_in_action_range) & cut_set:  # 若 action 裡有 frame 是分段點
            continue
        if IS_NORMAL:
            normalized_data = normalization(copy.deepcopy(np.array(data_arr[frame_in_action_range])))
        else:
            normalized_data = copy.deepcopy(np.array(data_arr[frame_in_action_range]))
        for frame_idx in range(N_STEPS):
            result_list.append(normalized_data[frame_idx])
        total_actions += 1

    return np.array(result_list, dtype=object), total_actions


# -----------------------------------------------------------
# 將座標點正規化 - 基準點為(0,0), frame 根據基準點來改變座標 &  弄成同一側(鏡像)/旋轉 & 魚的大小要一樣 (縮放)
# -----------------------------------------------------------
def normalization(data_arr):
    """
    將 data_arr 的座標點正規化: Affine transformation.
    Scale - 以 (0,0) 為基準點, frame1 縮放到 SCALE_LEN 大小.
    Translation - TRANSLATION_POINT 為基準點(0,0), frame 根據基準點來移動座標.
    Rotation - 以 (0,0) 為基準點, 順時針旋轉 n 度.
    Reflection - 以 y 軸為基準點, 弄成同一側(鏡像).

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(y, 3, 21-x)

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(y, 3, 21-x)
    """
    frame1 = data_arr[0]  # reference
    transformation_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if SCALE_LEN > 0:
        # 用 frame1 算出是否需縮放
        vec_th = np.array([frame1[0][MIRROR_POINT] - frame1[0][TRANSLATION_POINT],
                        frame1[1][MIRROR_POINT] - frame1[1][TRANSLATION_POINT]])  # tail - head
        fish1_len = np.linalg.norm(vec_th)
        if fish1_len != 0:
            scale = SCALE_LEN / fish1_len
            if scale != 1:
                scale_arr = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                # 先都縮放
                for idx in range(len(data_arr)):
                    data_arr[idx] = np.dot(scale_arr, data_arr[idx])
        else:
            raise ValueError('Failed: Not enough data')  # 照理來說不可能
    
    if IS_TRANSLATION:
        # 用 frame1 計算偏移矩陣
        [offset_x, offset_y] = 0-frame1[0][TRANSLATION_POINT], 0-frame1[1][TRANSLATION_POINT]
        if [offset_x, offset_y] != [0, 0]:
            offset_arr = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
            transformation_arr = np.dot(offset_arr, transformation_arr)

    if IS_ROTATE == 1:
        # 用 frame1 算出是否需旋轉
        [v2_x, v2_y] = frame1[0][MIRROR_POINT]-frame1[0][TRANSLATION_POINT], frame1[1][MIRROR_POINT]-frame1[1][TRANSLATION_POINT]
        if (not(v2_x > 0 and v2_y == 0)):  # 若 tail 不在 head 的正右邊
            [v1, v2] = np.array([1, 0]), np.array([v2_x, v2_y])
            [len_v1, len_v2] = np.linalg.norm(v1), np.linalg.norm(v2)

            sin_rho = np.cross(v2, v1) / (len_v1 * len_v2)
            cos_theta = np.dot(v2, v1) / (len_v1 * len_v2)

            rotate_arr = np.array([[cos_theta, -sin_rho, 0], [sin_rho, cos_theta, 0], [0, 0, 1]])
            transformation_arr = np.dot(rotate_arr, transformation_arr)
    elif IS_ROTATE == 2:
        # 用 frame1 算出是否需水平鏡像
        if frame1[0][MIRROR_POINT] - frame1[0][TRANSLATION_POINT] < 0:  # 若 tail_x 在 head_x 的左邊
            mirror_arr = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            transformation_arr = np.dot(mirror_arr, transformation_arr)

    if not np.array_equal(transformation_arr, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        for idx in range(len(data_arr)):
            data_arr[idx] = np.dot(transformation_arr, data_arr[idx])

    return data_arr


def _getENV():
    """
    取得環境變數

    Returns
    -------
    `dict` 10個元素
    """
    return {
        'project_path': PROJECT_PATH,
        'not_used_list': NOT_USED_LIST,
        'total_points': TOTAL_POINTS,
        'n_steps': N_STEPS,
        'check_front_back': CHECK_FRONT_BACK,
        'translation_point': TRANSLATION_POINT,
        'mirror_point': MIRROR_POINT,
        'is_rotate': IS_ROTATE,
        'scale_len': SCALE_LEN,
        'jump_n_frame':JUMP_N_FRAME,    
    }


# -----------------------------------------------------------
# The files converts to LSTM format.
# -----------------------------------------------------------
try:
    result = {'successful_times' : 0, 'failed_times' : 0}
    failed_path_list = []
    total_actions = {'train' : 0, 'validate' : 0, 'test' : 0}
    dirs = total_actions.keys()
    for dir in dirs:
        csv_path = PROJECT_PATH + '/TESTING/CSV/' + dir
        f_list = listdir(csv_path)
        if len(f_list) == 0:
            print("[警告] {} 沒有任何檔案".format(csv_path))
            continue
        else:
            csv_file_list = [f for f in f_list if isfile(join(csv_path, f))]
        for file_name in csv_file_list:
            _current_file_path = csv_path + '/' + file_name
            print("開始對 {} 進行轉換.....".format(file_name))

            X_data = getCoords(_current_file_path)
            X_usable_data, n_actions = convertToUseful(X_data)
            
            total_actions[dir] += n_actions

            if n_actions < 1 :
                failed_path_list.append(_current_file_path)
                result['failed_times'] += 1
                print("Failure！")
            else:
                if dir == 'test': # write X_usable_data into new file
                    label=0
                else: # write X_usable_data & label into new file
                    if(file_name[0]=='n'):
                        label=1
                    elif(file_name[0]=='f'):
                        label=2
                    elif(file_name[0]=='s'):
                        label=3
                    else:
                        label=4
                if writeData(PROJECT_PATH + '/TESTING/convertTo_txt', dir, X_usable_data, n_actions, label):
                    result['successful_times'] += 1
                    print("Success！")
                else:
                    print("Failure！")

    if result['failed_times'] > 0:
        print("{} 個檔案轉換成功 / {} 個檔案轉換失敗 : 失敗檔案如下".format(result['successful_times'], result['failed_times']))
        for path in failed_path_list:
            print("\t{}".format(path))
        FishLog.writeLog(FishLog.formatLog(30, "TESTING_convertData.py", "line 470", "List the fail files.", "failed files: {}".format(failed_path_list)))
    else:
        print("全部檔案皆轉換成功，請至 [ {}/TESTING/convertTo_txt/ ] 查看相關產出檔案".format(PROJECT_PATH))
    # -----------------------------------------------------------
    # 紀錄測試參數值
    # -----------------------------------------------------------
    record_dir = 'TESTING/records/'
    record_file_name = ''
    if not isdir(record_dir):
        makedirs(record_dir)

    with open(record_dir + '_result.txt', 'a') as f:
        f.write("[n_steps={:<2d} JUMP_N_FRAME={:<2d} is_normal={:<1d} scale_len={:<1d} is_translation={:<1d} is_rotate={:<1d}] data count={}\n".format(N_STEPS, JUMP_N_FRAME, IS_NORMAL, SCALE_LEN, IS_TRANSLATION, IS_ROTATE, total_actions))

    return_code = 0
    if total_actions['train']<1000 or total_actions['validate']<1:
        return_code = -1
    else:
        with open(record_dir + '_array.txt', 'a') as f:
            f.write("[n_steps={:<2d} JUMP_N_FRAME={:<2d} is_normal={:<1d} scale_len={:<1d} is_translation={:<1d} is_rotate={:<1d}]\n".format(N_STEPS, JUMP_N_FRAME, IS_NORMAL, SCALE_LEN, IS_TRANSLATION, IS_ROTATE))
except Exception as e:
    return_code = -1
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    FishLog.writeLog(FishLog.formatException(e.args[0], extract_tb(tb)[0], "Convert data.", "current file={}".format(_current_file_path)))
else:
    # log
    FishLog.writeLog(FishLog.formatLog(20, "TESTING_convertData.py", "line 496", "Convert data.", "[data count={} result={}] [n_steps={:<2d} JUMP_N_FRAME={:<2d} is_normal={:<1d} scale_len={:<1d} is_rotate={:<1d}]".format(total_actions, result, N_STEPS, JUMP_N_FRAME, IS_NORMAL, SCALE_LEN, IS_ROTATE)))
    sys.exit(return_code)