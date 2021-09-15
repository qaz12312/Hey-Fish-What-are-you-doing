import sys
# adding Folder_2 to the system path
sys.path.insert(0, 'C:/Users/88692/Desktop/Hey-Fish-What-are-you-doing')
import unittest
import convertData
import numpy as np
import copy
import FishDebug


class TestConvertData(unittest.TestCase):
    def setUp(self):
        self.data_arr = np.array([
            [[7.5, 7.0, 7.0, 8.0, 7.0, 7.0, 6.0], [3.0, 3.0, 4.0, 5.5, 5.5, 7.0, 5.5], [1, 1, 1, 1, 1, 1, 1]],  # green
            [[5.5, 5.0, 5.0, 6.0, 5.0, 5.0, 4.0], [4.0, 4.0, 5.0, 6.5, 6.5, 8.0, 6.5], [1, 1, 1, 1, 1, 1, 1]],  # green + (-2, 1)
            [[8.0, 8.0, 7.0, 5.5, 5.5, 4.0, 5.5], [2.5, 3.0, 3.0, 2.0, 3.0, 3.0, 4.0], [1, 1, 1, 1, 1, 1, 1]],  # yellow
            [[0.0, 0.0, 1.0, 2.5, 2.5, 4.0, 2.5], [1.5, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0], [1, 1, 1, 1, 1, 1, 1]],  # black
            [[2.5, 2.0, 2.0, 3.0, 2.0, 2.0, 1.0], [8.0, 8.0, 7.0, 5.5, 5.5, 4.0, 5.5], [1, 1, 1, 1, 1, 1, 1]],  # red
            [[3.5, 3.0, 3.0, 4.0, 3.0, 3.0, 2.0], [8.0, 8.0, 7.0, 5.5, 5.5, 4.0, 5.5], [1, 1, 1, 1, 1, 1, 1]],  # red + (1, 0)
        ], dtype=object)

        self.missing_arr = np.array([
            [[1, 2, None],   [4, 5, None],    [7, 8, None]],     # n
            [[6, 7, None],   [9, 10, None],   [12, 13, None]],   # n+5
            [[8, None, 12],  [14., None, 18], [20, None, 24]],   # (n+3)*2
            [[None, 13, 14], [None, 21, 22],  [None, 24, 26]],   # (n+6)//5*10+(n+6)%5
            [[None, 28, 32], [None, 40, 44],  [None, 52, 56]],   # (n+5)/4*16
            [[40, None, 60], [70, None, 90],  [100, None, 120]], # (n+3)/2*20
        ], dtype=object)
        for i in range(len(self.missing_arr)):
            self.missing_arr[i] = self.missing_arr[i].astype(np.float64)

    def tearDown(self):
        pass

    def test_getCoords(self):
        print_data = {'use data_arr': 'tests/Unit/DLC_log.csv'}
        expected = {
            'shape': (19, 3, 21-8),
            'z_axis': 21 - 8
        }

        result = convertData.getCoords('tests/Unit/DLC_log.csv')

        self.assertIsInstance(result, type(np.array([])))
        self.assertEqual(expected['shape'], result.shape)
        for idx in range(len(result)):
            frame = result[idx]
            self.assertEqual(expected['z_axis'], np.sum(frame[2]))
            print_data[idx] = "[{},{}],".format(list(frame[0]), list(frame[1]))

        FishDebug.writeLog(
            {"lineNum": 43,
             "funName": "test_getCoords",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "convertData/unittest/1getCoords.log",
            print_data
        )

    def test_writeData(self):
        result = convertData.writeData('debug/convertData/unittest/0writeData.txt', self.data_arr)

        self.assertTrue(result)
        # 要去確認文件

    def test_convertToUseful(self):
        CHECK_FRONT_BACK = [0, 2]
        print_data = {'use data_arr': 'missing_arr'}

        # --function content----------------------------------
        cut_list = []            # 要註記的 frame_number (從 0 開始算)
        interpolation_dict = {}  # { frame_number : [要填補的座標點] }
        for frame_idx in range(len(self.missing_arr)):
            print_data_item = []
            empty_indices_arr = (np.where(self.missing_arr[frame_idx][0] != self.missing_arr[frame_idx][0]))[0]  # 用 frame 的 x 座標找出值 = nan 的座標
            print_data_item.append("empty_indices_arr = {}".format(empty_indices_arr))

            # 若 frame 是正背面 - 正: [背鰭(後), 尾巴(前偏上), 尾巴(前偏下)] / 背: [嘴, 背鰭(前), 腹鰭(前), 比腹鰭再前面的一點]
            is_front_back_arr = np.in1d(CHECK_FRONT_BACK, empty_indices_arr, assume_unique=True)
            print_data_item.append("is_front_back_arr = {}".format(is_front_back_arr))
            if (np.sum(is_front_back_arr[:3]) == 3) or (np.sum(is_front_back_arr[3:]) == 4):
                cut_list.append(frame_idx)
            else:
                # 若 frame 是左右側 - 左: [右眼] / 右: [左眼]
                is_side_arr = np.in1d([0, 1], empty_indices_arr)
                print_data_item.append("is_side_arr = {}".format(is_side_arr))
                for coord_idx in range(2):
                    if is_side_arr[coord_idx]:
                        copy_idx = 0 if coord_idx == 1 else 1
                        self.missing_arr[frame_idx][0][coord_idx] = self.missing_arr[frame_idx][0][copy_idx]
                        self.missing_arr[frame_idx][1][coord_idx] = self.missing_arr[frame_idx][1][copy_idx]
                # 若仍有座標 = nan
                empty_indices_arr = (np.where(self.missing_arr[frame_idx][0] != self.missing_arr[frame_idx][0]))[0]
                print_data_item.append("still has empty_indices_arr = {}".format(empty_indices_arr))
                if empty_indices_arr.size > 0:
                    if frame_idx > 1:  # 因期望值是2次多項式
                        interpolation_dict[frame_idx] = empty_indices_arr
                    else:
                        cut_list.append(frame_idx)
            print_data["frame {}".format(frame_idx)] = print_data_item

        print_data["\ninterpolation_dict"] = "{}".format(interpolation_dict)
        print_data["cut_list"] = "{}".format(cut_list)
        print_data_item = []
        for idx in range(len(self.missing_arr)):
            frame = self.missing_arr[idx]
            print_data_item.append("\n[{},{},{}],".format(list(frame[0]), list(frame[1]), list(frame[2])))
        print_data["\nresult data_arr"] = print_data_item
        FishDebug.writeLog(
            {"lineNum": 70,
             "funName": "test_convertToUseful",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "convertData/unittest/2convertToUseful.log",
            print_data
        )

    def test__fillInCoords(self):
        print_data = {'use data_arr': 'missing_arr'}
        test_data = [
            {},
            {0: [2], 1: [2], 2: [1], 3: [0], 4: [0], 5: [1]},
        ]
        expected = {
            'shape': self.missing_arr.shape,
            'z_axis': [
                [7, 8, np.newaxis],
                [12, 13, np.newaxis],
                [20, np.newaxis, 24],
                [np.newaxis, 24, 26],
                [np.newaxis, 52, 56],
                [100, np.newaxis, 120]
            ]
        }

        for test_case in range(len(test_data)):
            print_data["Test case {}".format(test_case)] = "interpolation_dict = {}".format(test_data[test_case])

            result = convertData._fillInCoords(self.missing_arr, test_data[test_case])

            self.assertIsInstance(result, type(np.array([])))
            self.assertEqual(expected['shape'], result.shape)
            print_data_item = []
            for idx in range(len(result)):
                frame = result[idx]
                self.assertEqual(len(expected['z_axis'][idx]), len(frame[2]))
                for z_idx in range(len(frame[2])):
                    if expected['z_axis'][idx][z_idx] == None:
                        self.assertTrue(frame[2][z_idx] != frame[2][z_idx])
                    else:
                        self.assertEqual(expected['z_axis'][idx][z_idx], frame[2][z_idx])
                print_data_item.append("[{},{}],".format(list(frame[0]), list(frame[1])))
                print_data["result{} data_arr".format(test_case)] = print_data_item

        FishDebug.writeLog(
            {"lineNum": 138,
             "funName": "test__fillInCoords",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "convertData/unittest/3_fillInCoords.log",
            print_data
        )

    def test__frameSplit(self):
        cut_list = [5]
        N_STEPS = 3
        CHECK_ROTATE = True
        print_data = {'use data_arr': 'data_arr'}

        # --function content----------------------------------
        result_list = []
        cut_set = set(cut_list)
        max_action_count = len(self.data_arr)-N_STEPS+1  # 最多會有 y 個 actions 產生
        print_data["The actions that will be at most generated"] = max_action_count

        for start_frame_idx in range(max_action_count):
            frame_in_action_range = range(start_frame_idx, start_frame_idx+N_STEPS)
            if set(frame_in_action_range) & cut_set:  # 若 action 裡有 frame 是分段點
                print_data["{}".format(set(frame_in_action_range))] = "No"
                continue

            need_to_normalize = np.array(self.data_arr[frame_in_action_range])
            
            print_data_item = []
            for idx in range(len(need_to_normalize)):
                frame = need_to_normalize[idx]
                print_data_item.append("\t[{},{},{}],".format(list(frame[0]), list(frame[1]), list(frame[2])))
            print_data["\nframe{} need_to_normalize".format(set(frame_in_action_range))] = print_data_item

            normalized_data = convertData.normalization(copy.deepcopy(need_to_normalize), CHECK_ROTATE)

            print_data_item = []
            for idx in range(len(normalized_data)):
                frame = normalized_data[idx]
                print_data_item.append("\t[{},{},{}],".format(list(frame[0]), list(frame[1]), list(frame[2])))
            print_data["frame{} normalized_data".format(set(frame_in_action_range))] = print_data_item
            print_data["frame{} shape of normalized_data / need_to_normalize".format(set(frame_in_action_range))] = "{}/{}".format(normalized_data.shape, need_to_normalize.shape)

            for frame_idx in range(N_STEPS):
                result_list.append(normalized_data[frame_idx])

        res = np.array(result_list, dtype=object)
        print_data["result shape"] = res.shape
        print_data_item = []
        for idx in range(len(res)):
            frame = res[idx]
            print_data_item.append("[{},{},{}],".format(list(frame[0]), list(frame[1]), list(frame[2])))
        print_data["result data_arr"] = print_data_item
        FishDebug.writeLog(
            {"lineNum": 168,
             "funName": "test__frameSplit",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "convertData/unittest/4_frameSplit.log",
            print_data
        )

    def test_normalization(self):
        print_data = {'use data_arr': 'data_arr'}
        expected = {
            'shape': self.data_arr.shape,
            'z_axis': copy.deepcopy(self.data_arr[0][2])
        }

        for check_rotate in [True, False]:
            print_data["Test case {} check_rotate".format(int(check_rotate))] = str(check_rotate)

            result = convertData.normalization(self.data_arr, check_rotate)

            self.assertIsInstance(result, type(np.array([])))
            self.assertEqual(expected['shape'], result.shape)
            print_data_item = []
            for idx in range(len(result)):
                frame = result[idx]
                self.assertEqual(len(expected['z_axis']), len(frame[2]))
                for z_idx in range(len(frame[2])):
                    self.assertEqual(expected['z_axis'][z_idx], frame[2][z_idx])
                print_data_item.append("[{},{}],".format(list(frame[0]), list(frame[1])))  # 需要畫出圖才能確定
            print_data["Test case {} result data_arr".format(int(check_rotate))] = print_data_item

        FishDebug.writeLog(
            {"lineNum": 225,
             "funName": "test_normalization",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "convertData/unittest/5normalization.log",
            print_data
        )


if __name__ == '__main__':
    unittest.main()
