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
            [[None, 2, 3, 4, None, 6, 7, 8, 9, 10], [None, 4, 6, 8, None, 12, 14, 16, 18, 20], [None, 0, 0, 0, None, 0, 0, 0, 0, 0]],  # n
            [[6, 7, None, 9, 10, 11, None, None, 14, 15],  [7, 9, None, 13, 15, 17, None, None, 23, 25], [1, 1, None, 1, 1, 1, None, None, 1, 1]],  # (n+7)*2
            [[16, 18, 20, None, None, None, 28, 30, 32, 34],  [18, 22, 26, None, None, None, 42, 46, 50, 54], [2, 2, 2, None, None, None, 2, 2, 2, 2]],  # (n+11)//5*10+n%5
            [[None, None, 23, 34, 30, 31, 32, None, 44, 40], [None, None, 31, 33, 40, 42, 54, None, 53, 60], [None, None, 3, 3, 3, 3, 3, None, 3, 3]],  # (n+6)//5*10+(n+6)%5
            [[56, 60, 64, 68, 72, None, 80, None, None, 92],  [60, 68, 76, 84, 92, None, 108, None, None, 132], [4, 4, 4, 4, 4, None, 4, None, None, 4]],  # (n+13)*4
            [[180, 190, 200, 210, 220, 230, 240, 250, None, None],  [190, 210, 230, 250, 270, 290, 310, 330, None, None], [5, 5, 5, 5, 5, 5, 5, 5, None, None]],  # (n+17)*10
        ], dtype=object)
        for i in range(len(self.missing_arr)):
            self.missing_arr[i] = self.missing_arr[i].astype(np.float64)

    def tearDown(self):
        pass

    def test_getCoords(self):
        log_data = {'The data_arr is use': 'tests/Unit/DLC_log.csv'}
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
            
            log_data["frame[{}] (x,y)".format(idx)] = "[\n\t{},\n\t{}],".format(list(frame[0]), list(frame[1]))
        FishDebug.writeLog(
            {"lineNum": 43,
             "funName": "test_getCoords",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/1getCoords.log",
            log_data
        )

    def test_writeData(self):
        test_data = np.array([
            [[1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14], [1, 1, 1, 1, 1, 1, 1]],
            [[10, 30, 50, 70, 90, 110, 130], [20, 40, 60, 80, 100, 120, 140], [1, 1, 1, 1, 1, 1, 1]],
        ], dtype=object)

        result = convertData.writeData('debug/_Unit_Test/convertData/0writeData.txt', test_data)

        self.assertTrue(result)
        # 要去確認文件
    
    def test_convertToUsefulPart1(self):
        log_data = {'The data_arr is use': 'missing_arr'}
        test_case = [[], [2], [5, 7, 8]]  # CHECK_FRONT_BACK

        for check_front_back in test_case:
            log_data_item = {}
            data_arr = copy.deepcopy(self.missing_arr)

            # --function content----------------------------------
            cut_list = []
            interpolation_dict = {}
            for frame_idx in range(len(data_arr)):
                empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]
                is_front_back_arr = np.in1d(check_front_back, empty_indices_arr, assume_unique=True)

                log_data_item_frame = {
                    "empty coord_idx": empty_indices_arr,
                    "is front back {}".format(set(check_front_back)): is_front_back_arr}

                if (np.sum(is_front_back_arr[:3]) == 3) or (np.sum(is_front_back_arr[3:]) == 4):
                    cut_list.append(frame_idx)
                else:
                    is_side_arr = np.in1d([0, 1], empty_indices_arr)
                    for coord_idx in range(2):
                        if is_side_arr[coord_idx]:
                            copy_idx = 0 if coord_idx == 1 else 1
                            data_arr[frame_idx][0][coord_idx] = data_arr[frame_idx][0][copy_idx]
                            data_arr[frame_idx][1][coord_idx] = data_arr[frame_idx][1][copy_idx]
                    empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]
                    if empty_indices_arr.size > 0:
                        if frame_idx > 1:
                            interpolation_dict[frame_idx] = empty_indices_arr
                        else:
                            cut_list.append(frame_idx)

                    log_data_item_frame["is side {0,1}"] = is_side_arr
                    log_data_item_frame["still has empty coord_idx"] = empty_indices_arr
                log_data_item["frame[{}]".format(frame_idx)] = log_data_item_frame
            log_data_item["interpolation_dict for _fillInCoords()"] = interpolation_dict
            log_data_item["cut_list for _frameSplit()"] = cut_list
            log_data_item["result data_arr"] = data_arr
            log_data["Test case-CHECK_FRONT_BACK={}".format(check_front_back)] = log_data_item
        FishDebug.writeLog(
            {"lineNum": 79,
             "funName": "test_convertToUsefulPart1",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/21convertToUseful.log",
            log_data
        )

    def test_convertToUsefulPart2(self):
        log_data = {'The data_arr is use': 'missing_arr'}
        test_case = [[], [2], [5, 7, 8]]  # CHECK_FRONT_BACK

        for check_front_back in test_case:
            data_arr = copy.deepcopy(self.missing_arr)

            # --function content----------------------------------
            cut_list = []
            interpolation_dict = {}
            for frame_idx in range(len(data_arr)):
                empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]
                is_front_back_arr = np.in1d(check_front_back, empty_indices_arr, assume_unique=True)
                if (np.sum(is_front_back_arr[:3]) == 3) or (np.sum(is_front_back_arr[3:]) == 4):
                    cut_list.append(frame_idx)
                else:
                    is_side_arr = np.in1d([0, 1], empty_indices_arr)
                    for coord_idx in range(2):
                        if is_side_arr[coord_idx]:
                            copy_idx = 0 if coord_idx == 1 else 1
                            data_arr[frame_idx][0][coord_idx] = data_arr[frame_idx][0][copy_idx]
                            data_arr[frame_idx][1][coord_idx] = data_arr[frame_idx][1][copy_idx]
                    empty_indices_arr = (np.where(data_arr[frame_idx][0] != data_arr[frame_idx][0]))[0]
                    if empty_indices_arr.size > 0:
                        if frame_idx > 1:
                            interpolation_dict[frame_idx] = empty_indices_arr
                        else:
                            cut_list.append(frame_idx)
            if bool(interpolation_dict):
                data_arr = convertData._fillInCoords(data_arr, interpolation_dict)
            
            log_data["Test case-CHECK_FRONT_BACK={}".format(check_front_back)] = data_arr
        FishDebug.writeLog(
            {"lineNum": 128,
             "funName": "test_convertToUsefulPart2",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/22convertToUseful.log",
            log_data
        )

    def test__fillInCoords(self):
        log_data = {'The data_arr is use': 'missing_arr'}
        test_case = [  # interpolation_dict
            {}, 
            {0: [0, 4], 1: [2, 6, 7], 2: [3, 4, 5], 3: [0, 1, 7], 4: [5, 7, 8], 5: [8, 9]},
        ]
        expected = {
            'shape': self.missing_arr.shape,
            'z_axis': [
                [np.newaxis, 0, 0, 0, np.newaxis, 0, 0, 0, 0, 0],
                [1, 1, np.newaxis, 1, 1, 1, np.newaxis, np.newaxis, 1, 1],
                [2, 2, 2, np.newaxis, np.newaxis, np.newaxis, 2, 2, 2, 2],
                [np.newaxis, np.newaxis, 3, 3, 3, 3, 3, np.newaxis, 3, 3],
                [4, 4, 4, 4, 4, np.newaxis, 4, np.newaxis, np.newaxis, 4],
                [5, 5, 5, 5, 5, 5, 5, 5, np.newaxis, np.newaxis],
            ]
        }

        for interpolation_dict in test_case:
            data_arr = copy.deepcopy(self.missing_arr)

            result = convertData._fillInCoords(data_arr, interpolation_dict)

            self.assertIsInstance(result, type(np.array([])))
            self.assertEqual(expected['shape'], result.shape)
            for idx in range(len(result)):
                frame = result[idx]
                self.assertEqual(len(expected['z_axis'][idx]), len(frame[2]))
                for z_idx in range(len(frame[2])):
                    if expected['z_axis'][idx][z_idx] == None:
                        self.assertTrue(frame[2][z_idx] != frame[2][z_idx])
                    else:
                        self.assertEqual(expected['z_axis'][idx][z_idx], frame[2][z_idx])

            log_data["Test case: interpolation_dict={}".format(interpolation_dict)] = result
        FishDebug.writeLog(
            {"lineNum": 182,
             "funName": "test__fillInCoords",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/3_fillInCoords.log",
            log_data
        )

    def test__frameSplitPart(self):
        log_data = {'The data_arr is use': '0 ~ 149'}
        test_case = [
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':30, 'JUMP_N_FRAME':1},
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':30, 'JUMP_N_FRAME':5},
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':10, 'JUMP_N_FRAME':7},
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':18, 'JUMP_N_FRAME':8},
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':5, 'JUMP_N_FRAME':30},
            {'cut_list': [5, 21, 22, 23, 89, 123], 'N_STEPS':8, 'JUMP_N_FRAME':30},
        ]

        for test in test_case:
            data_arr = np.arange(150)  # 5 秒 150 frame
            cut_list = test['cut_list']
            N_STEPS = test['N_STEPS']
            JUMP_N_FRAME = test['JUMP_N_FRAME']

            # --function content----------------------------------
            result_list = []
            cut_set = set(cut_list)
            
            max_action_count = len(data_arr)-((N_STEPS-1)*JUMP_N_FRAME+1)+1
            log_data_item = {
                "The actions that will be at most generated": max_action_count}

            for start_frame_idx in range(max_action_count):
                frame_in_action_range = range(start_frame_idx, start_frame_idx+((N_STEPS-1)*JUMP_N_FRAME+1), JUMP_N_FRAME)
                if set(frame_in_action_range) & cut_set:
                    log_data_item["{}".format(tuple(frame_in_action_range))] = "No"
                    continue
                log_data_item["{}".format(tuple(frame_in_action_range))] = "Yes"
                normalized_data = copy.deepcopy(np.array(data_arr[frame_in_action_range]))
                for frame_idx in range(N_STEPS):
                    result_list.append(normalized_data[frame_idx])
            
            log_data_item["result data_arr"] = result_list
            result = np.array(result_list, dtype=object)
            log_data_item["result shape"] = result.shape
            log_data["Test case-{}".format(test)] = log_data_item
        FishDebug.writeLog(
            {"lineNum": 221,
             "funName": "test__frameSplitPart",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/41_frameSplit.log",
            log_data
        )

    def test__frameSplit(self):
        log_data = {'The data_arr is use': 'data_arr'}
        test_case = [
            {'cut_list': [5], 'N_STEPS':3, 'JUMP_N_FRAME':1, 'CHECK_ROTATE':True},
        ]

        for test in test_case:
            log_data_item = {}
            data_arr = copy.deepcopy(self.data_arr)
            cut_list = test['cut_list']
            N_STEPS = test['N_STEPS']
            JUMP_N_FRAME = test['JUMP_N_FRAME']
            CHECK_ROTATE = test['CHECK_ROTATE']

            # --function content----------------------------------
            result_list = []
            cut_set = set(cut_list)

            max_action_count = len(data_arr)-((N_STEPS-1) * JUMP_N_FRAME+1)+1
            for start_frame_idx in range(max_action_count):
                frame_in_action_range = range(start_frame_idx, start_frame_idx+((N_STEPS-1)*JUMP_N_FRAME+1), JUMP_N_FRAME)
                if set(frame_in_action_range) & cut_set:
                    continue
                need_to_normalize = np.array(self.data_arr[frame_in_action_range])
                normalized_data = convertData.normalization(copy.deepcopy(need_to_normalize), CHECK_ROTATE)
                for frame_idx in range(N_STEPS):
                    result_list.append(normalized_data[frame_idx])
                
                log_data_item["frame{}".format(list(frame_in_action_range))] = {
                    "need_to_normalize": need_to_normalize,
                    "normalized_data": normalized_data,
                    "shape of need_to_normalize / normalized_data": "{}/{}".format(need_to_normalize.shape, normalized_data.shape),
                }
            result = np.array(result_list, dtype=object)

            log_data_item["result data_arr"] = result
            log_data_item["result shape"] = result.shape
            log_data["Test case-{}".format(test)] = log_data_item
        FishDebug.writeLog(
            {"lineNum": 265,
             "funName": "test__frameSplit",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/4_frameSplit.log",
            log_data
        )

    def test_normalization(self):
        log_data = {'The data_arr is use': 'data_arr'}
        test_case = [True, False]  # CHECK_ROTATE
        expected = {
            'shape': self.data_arr.shape,
            'z_axis': copy.deepcopy(self.data_arr[0][2])
        }

        for check_rotate in test_case:
            data_arr = copy.deepcopy(self.data_arr)
            
            result = convertData.normalization(data_arr, check_rotate)

            self.assertIsInstance(result, type(np.array([])))
            self.assertEqual(expected['shape'], result.shape)
            log_data_item = []
            for idx in range(len(result)):
                frame = result[idx]
                self.assertEqual(len(expected['z_axis']), len(frame[2]))
                for z_idx in range(len(frame[2])):
                    self.assertEqual(expected['z_axis'][z_idx], frame[2][z_idx])
                
                log_data_item.append("[{},{}],".format(list(frame[0]), list(frame[1])))  # 需要畫出圖才能確定
            log_data["Test case-check_rotate={}".format(str(check_rotate))] = log_data_item
        FishDebug.writeLog(
            {"lineNum": 308,
             "funName": "test_normalization",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/5normalization.log",
            log_data
        )

    def test__getENV(self):
        result = convertData._getENV()

        self.assertIsInstance(result, type({}))
        self.assertEqual(10, len(result))

        FishDebug.writeLog(
            {"lineNum": 330,
             "funName": "test__getENV",
             "fileName": "./tests/Unit/TestConvertData.py"},
            "_Unit_Test/convertData/6_getENV.log",
            result
        )


if __name__ == '__main__':
    unittest.main()
