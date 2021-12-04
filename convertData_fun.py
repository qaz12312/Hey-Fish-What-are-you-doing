"""
將 convertData 裡的 code 拆成獨立 function
"""
import numpy as np


def frameScale(p_head, p_tail, SCALE_LEN):
    """
    求縮放矩陣
        frame 中, p_head 到 p_tail 的長度縮放到 SCALE_LEN 大小.
        以 (0,0) 為基準點進行縮放.

    Parameters
    ----------
    p_head : `dict`
        frame 的座標點 {x,y}
    p_tail : `dict`
        frame 的座標點 {x,y}
    SCALE_LEN : `float`
        縮放長度

    Returns
    -------
    need_scale : `bool`
        是否需縮放
    scale_arr : `ndarray`
        If need_scale = False, scale_arr = [].
    """
    vector_t2h = np.array([p_tail['x'] - p_head['x'],
                           p_tail['y'] - p_head['y']])  # tail - head
    fish_len = np.linalg.norm(vector_t2h)
    if fish_len != 0:
        scale = SCALE_LEN / fish_len
        if scale != 1:
            return True, np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ])
    else:
        raise ValueError('Failed: Not enough data')  # 照理來說不可能

    return False, np.array([])


def frameTranslation(x, y):
    """
    求偏移矩陣
        將 (x,y) 設為基準點 (0,0), frame 根據基準點來移動座標.

    Parameters
    ----------
    x : `float`
        frame 的 p 的 x 座標值
    y : `float`
        frame 的 p 的 y 座標值

    Returns
    -------
    need_offset : `bool`
        是否需偏移
    offset_arr : `ndarray`
        If need_offset = False, offset_arr = [].
    """
    [offset_x, offset_y] = [0-x, 0-y]
    if [offset_x, offset_y] != [0, 0]:
        return True, np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ])

    return False, np.array([])


def frameRotate(p_head, p_tail):
    """
    求旋轉矩陣 
        frame 中, p_head 到 p_tail 的向量與 v1(1,0,0) 夾的角.
        以 (0,0) 為基準點進行順時針旋轉 ( v2 相對 v1 的角度 ).

    Parameters
    ----------
    p_head : `dict`
        frame 的座標點 {x,y}
    p_tail : `dict`
        frame 的座標點 {x,y}

    Returns
    -------
    need_rotate : `bool`
        是否需旋轉
    rotate_arr : `ndarray`
        If need_rotate = False, rotate_arr = [].
    """
    v2_x = p_tail['x'] - p_head['x']
    v2_y = p_tail['y'] - p_head['y']
    if (not(v2_x > 0 and v2_y == 0)):  # 若 tail 不在 head 的正右邊
        [v1, v2] = np.array([1, 0]), np.array([v2_x, v2_y])
        [len_v1, len_v2] = np.linalg.norm(v1), np.linalg.norm(v2)

        sin_rho = np.cross(v2, v1) / (len_v1 * len_v2)
        cos_theta = np.dot(v2, v1) / (len_v1 * len_v2)

        return True, np.array([
            [cos_theta, -sin_rho, 0],
            [sin_rho, cos_theta, 0],
            [0, 0, 1]
        ])

    return False, np.array([])


def frameMirror(head_x, tail_x):
    """
    求鏡像矩陣
        用 frame 中, p1 跟 p2 的位置, 判斷是否需要水平鏡像
        以 y 軸為基準點, 弄成同一側(鏡像).

    Parameters
    ----------
    head_x : `float`
        frame 的 p1 的 x 座標值
    tail_x : `float`
        frame 的 p2 的 x 座標值

    Returns
    -------
    need_mirror : `bool`
        是否需要水平鏡像
    mirror_arr : `ndarray`
        If need_mirror = False, mirror_arr = [].
    """
    if tail_x - head_x < 0:  # 若 tail_x 在 head_x 的左邊
        return True, np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    return False, np.array([])
