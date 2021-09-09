"""
將 Deeplabcut 的 log 檔中的資料轉成 LSTM 所需格式.
"""
from csv import reader
import numpy as np
from os import getenv, listdir
from os.path import isfile, join
from dotenv import load_dotenv
import FishDebug

load_dotenv()
PROJECT_PATH = getenv('PROJECT_PATH')
TOTAL_POINTS = int(getenv('TOTAL_POINTS'))           # 1 個 frame 有 n 個座標點
TRANSLATION_POINT = int(getenv('TRANSLATION_POINT')) # translation, 以第 n 個座標點為基準點(0,0)
MIRROR_POINT = int(getenv('MIRROR_POINT'))           # mirror, 以第 n 個座標點是否為負來判斷是否鏡像


# -----------------------------------------------------------
# Read coordinates from Deeplabcut's log file (.csv)
# -----------------------------------------------------------
def readFile(filePath):
    """
    Read coordinates from Deeplabcut's log file (.csv).

    Parameters
    ----------
    filePath : `str`
        The absolute path of the log file.

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D, 每一 dim 的元素個數 = [N(frame數), 3(x,y,z座標), 21(座標點)].
    """
    data_x = []

    with open(filePath, newline='') as f:
        frame_list = reader(f)  # 讀取 CSV 檔案內容
        line = 0
        for frame_coords in frame_list:
            line += 1
            if line < 4 : # 略過前3行
                continue
             
            temp_list = np.array(frame_coords[1:], dtype=object) # 第一個(xxx.png)不需要
            # 若有座標點 = ''
            empty_indices_arr = (np.where(temp_list == ''))[0]
            if empty_indices_arr.size > 0 :
                # 將 '' 轉成 None
                temp_list[empty_indices_arr] = np.newaxis
            # 轉成 float64 ndarray
            temp_list = temp_list.astype(np.float64)
            
            data_x.append([
                temp_list[::2],    # x 座標
                temp_list[1::2],   # y 座標
                [1] * TOTAL_POINTS # z 座標
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
def writeData(filePath, data_arr):
    """
    Write data(3D coordinate array) into file (.txt).

    Parameters
    ----------
    filePath : `str`
        The absolute path of the file.
    data_arr : `ndarray` (dtype = object, element type = float64)  /?是否只要是ndarray
        3D(n, 3, 21)

    Returns
    -------
    `True`
    """
    with open(filePath, 'w') as f:
        for frame in data_arr:
            # frame_arr = [x1, y1, ..., xn, yn]
            frame_arr = frame[0][:] # frame_arr = x 座標 array
            for idx in range(len(frame[1])): # 將 y 座標 array insert 到 frame_arr
                frame_arr = np.insert(frame_arr, idx*2+1, frame[1][idx])
            # 取到小數點幾位?
            f.write("{}\n".format(",".join(x for x in ["%s" % num for num in frame_arr])))
    
    return True


# -----------------------------------------------------------
# 移除正面、背面的 frame & 若為側面, 則將另一側的座標點複製過來 & 若有空資料，則用內插法補上
# -----------------------------------------------------------
def convertToUseful(data_arr):
    """
    移除正面、背面的 frame & 若為側面, 則將另一側的座標點複製過來 & 若有空資料，則用內插法補上.

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(N, 3, 21)

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(N-不能用的, 3, 21)
    """
    """Debug
    size = len(data_arr)
    arr = []
    # 前後, 左右, 前, 後, 左, 右, 成功
    times = [0, 0, 0, 0, 0, 0, 0]
    # 都缺, 缺3面, 缺2面, 缺1面, 全部座標都有
    fit_times = [0, 0, 0, 0, 0]
    for idx in range(size):
        empty_indices        = (np.where(data_arr[idx][0] != data_arr[idx][0]))[0]
        check_side       = np.in1d([0,1,2,3,4,5,6,7,8,9], empty_indices)
        check_front_back = np.in1d([13,14,17,10,11,18,20], empty_indices)
        check_need       = np.in1d([12,15,16,19], empty_indices)
        is_front             = np.sum(check_front_back[:3])
        is_back              = np.sum(check_front_back[3:])
        is_left              = np.sum(check_side[5:])
        is_right             = np.sum(check_side[:5])
        is_success           = np.sum(check_need)==0 # 必須有 
        
        total_fit = np.sum([(is_front > 0), (is_back > 1), (is_left > 0), (is_right > 0)])
        if total_fit == 4 :
            fit_times[0] += 1
        elif total_fit == 3 :
            fit_times[1] += 1
        elif total_fit == 2 :
            fit_times[2] += 1
            if is_front > 0 and is_back > 1 :
                times[0] += 1
            elif is_left and is_right :
                times[1] += 1
        elif total_fit > 1:
            fit_times[3] += 1
            if is_front > 0 :
                times[-5] += 1
            elif is_back > 1 :
                times[-4] += 1
            elif is_left:
                times[-3] += 1
            elif is_right:
                times[-2] += 1
        if is_success:
            times[-1] += 1

        if empty_indices.size == 0:
            fit_times[-1] += 1
            dict_item = {
                "empty" : empty_indices,
                "ONLY VALUE" : "Success" if is_success else "Failure",
            }
        else :
            dict_item = {
                "empty"    : empty_indices,
                "chk_f_b " : check_front_back,
                "front"    : is_front,
                "back "    : is_back,
                "chk_side" : check_side,
                "right"    : is_right,
                "left "    : is_left,
                "ONLY VALUE" : "Success" if is_success else "Failure",
            }
        arr.append(dict_item)
    dict_item = {
            "check_front_back" : [13,14,17,10,11,18,20],
            "front           " : [13,14,17],
            "back            " : [10,11,18,20],
            "check_side      " : [0,1,2,3,4,5,6,7,8,9],
            "right           " : [0,1,2,3,4],
            "left            " : [5,6,7,8,9],
            "should have     " : [12,15,16,19],
            "coords[lack 4, lack 3, lack 2, lack 1, all]" : "[{}, {}, {}, {}, {}]".format(fit_times[0],fit_times[1],fit_times[2],fit_times[3],fit_times[4]),
            "lack[f_b, l_r, f, b, l, r]" : "[{}, {}, {}, {}, {}, {}]".format(times[0],times[1],times[2],times[3],times[4],times[5]),
            "Success rate " : "{}/{}".format(times[-1], size),
        }
    arr.insert(0,dict_item)
    return arr
    """
    delete_list = []        # 要刪掉的 frame 的 index
    interpolation_dict = {} # 要內插的座標 & 其 frame number

    for idx in range(len(data_arr)):
        empty_indices = (np.where(data_arr[idx][0] != data_arr[idx][0]))[0] # 用 frame 的 x 座標找出 值 = nan 的座標

        check_front_back_arr = np.in1d([13,14,17,10,11,18,20], empty_indices)
        # 正: [背鰭(後), 尾巴(前偏上), 尾巴(前偏下)]
        # 背: [嘴, 背鰭(前), 腹鰭(前), 比腹鰭再前面一點]
        if (np.sum(check_front_back_arr[:3]) > 0) or (np.sum(check_front_back_arr[3:]) > 1): # 若 frame 是正面 or 背面
            delete_list.append(idx)
        else:
            check_side_arr = np.in1d([0,1,2,3,4,5,6,7,8,9], empty_indices)
            for index  in range(len(check_side_arr)):
                if check_side_arr[index] :  # 若 frame 的側面座標為空，則複製另一面的
                    copy_idx = (index + 5) % 10
                    data_arr[idx][0][index] = data_arr[idx][0][copy_idx]
                    data_arr[idx][1][index] = data_arr[idx][1][copy_idx]
            # elif np.sum(np.in1d([0,1,2,3,4], empty_indices)) > 0: # 若 frame 是右側
            #     for index in empty_indices:
            #         data_arr[idx][0][index] = data_arr[idx][0][index+5]
            #         data_arr[idx][1][index] = data_arr[idx][1][index+5]
            # elif np.sum(np.in1d([5,6,7,8,9], empty_indices)) > 0: # 若 frame 是左側
            #     for index in empty_indices:
            #         data_arr[idx][0][index] = data_arr[idx][0][index-5]
            #         data_arr[idx][1][index] = data_arr[idx][1][index-5]

            # 若仍有座標 = nan
            empty_indices_arr = (np.where(data_arr[idx][0] != data_arr[idx][0]))[0]
            if empty_indices_arr.size > 0 :
                interpolation_dict[idx] = empty_indices_arr
   
    if bool(interpolation_dict): # if interpolation_dict is not empty
        # print(interpolation_dict)
        data_arr = fillData(data_arr, interpolation_dict)

    return np.delete(data_arr, delete_list, axis=0) # Remove


# -----------------------------------------------------------
# 將座標點補齊 - 用線性插值
# -----------------------------------------------------------
def fillData(data_arr, interpolation_dict):
    """
    將座標點補齊 - 使用內插法.

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(n, 3, 21)
    interpolation_dict : `dict[int, 1D int ndarray]`
        有缺的座標點 & 其 frame number (`{..., {frameNum} = [coord_n,...], ...}`)
    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(n, 3, 21)
    """
    total_frames = len(data_arr)
    for frame_k,coords in interpolation_dict.items():
        for index in coords:
            value_py = []
            value_px = []
            frame_p = []

            up = frame_k + 1
            low = frame_k - 1
            while len(frame_p) < 3:
                if up < total_frames:
                    if data_arr[up,0,index] == data_arr[up,0,index]:
                        value_px.append(data_arr[up,0,index])
                        value_py.append(data_arr[up,1,index])

                        frame_p.append(up)
                    up += 1
                if low >= 0:
                    if data_arr[low,0,index] == data_arr[low,0,index]:
                        value_px.append(data_arr[low,0,index])
                        value_py.append(data_arr[low,1,index])

                        frame_p.append(low)
                    low -= 1

            data_arr[frame_k,0,index] = np.interp(frame_k, frame_p, value_px)
            data_arr[frame_k,1,index] = np.interp(frame_k, frame_p, value_py)

    return data_arr


# -----------------------------------------------------------
# 將座標點正規化 - 基準點為(0,0)，frame 根據基準點來改變座標 &  弄成同一側(鏡像)
# issue: 魚的大小要一樣 (縮放)
# -----------------------------------------------------------
def normalization(data_arr):
    """
    將座標點正規化 - 基準點為(0,0)，frame 根據基準點來改變座標 &  弄成同一側(鏡像).

    Parameters
    ----------
    data_arr : `ndarray` (dtype = object, element type = float64)
        3D(n, 3, 21)

    Returns
    -------
    `ndarray` (dtype = object, element type = float64)
        3D(n, 3, 21)
    """
    # 用 frame1 計算偏移矩陣
    frame1 = data_arr[0]
    [offset_x, offset_y] = [0-frame1[0][TRANSLATION_POINT], 0-frame1[1][TRANSLATION_POINT]]
    offset_arr = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

    # 用 frame1 算出是否需水平鏡像
    need_mirror = False
    if frame1[0][MIRROR_POINT] - frame1[0][TRANSLATION_POINT] < 0 : # 若 tail_x - head_x < 0
        mirror_arr = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        need_mirror = True
    
    # 每個 frame 做平移、水平鏡像 (先檢查是否需要內插)
    if [offset_x, offset_y] != [0,0] and need_mirror:
        for idx in range(len(data_arr)):
            res_arr = np.dot(offset_arr, data_arr[idx])
            data_arr[idx] = np.dot(mirror_arr, res_arr)
    elif [offset_x, offset_y] != [0,0]:
        for idx in range(len(data_arr)):
            data_arr[idx] = np.dot(offset_arr, data_arr[idx])
    elif need_mirror:
        for idx in range(len(data_arr)):
            data_arr[idx] = np.dot(mirror_arr, data_arr[idx])
    
    return data_arr


if __name__ == '__main__':
    print("test.")

    # The file names that need to be converted to LSTM format
    CSVpath = PROJECT_PATH + '/testCSV'
    # csvfiles = [f for f in listdir(CSVpath) if isfile(join(CSVpath, f))]
    csvfiles = ['fish_1.csv', 'fish_2.csv', 'fish_3.csv']
    for file in csvfiles:
        print(file)
        data_x = readFile(PROJECT_PATH + '/testCSV/' + file)
        usable_data_x = convertToUseful(data_x)
        normalized_data = normalization(usable_data_x)
        # write into new file
        status = "finish" if writeData(PROJECT_PATH + '/convertTo_txt/' + file[:-4] + '_X_train.txt', normalized_data) else "failure"
        print('./convertTo_txt/' + file + ': {}.\n'.format(status))
    
    print("finish.")