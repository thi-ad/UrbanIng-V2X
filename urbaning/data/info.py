crossing_cameras = {
    'crossing1': [
        'crossing1_13_thermal_camera',
        'crossing1_14_thermal_camera',
        'crossing1_15_thermal_camera',
        'crossing1_33_thermal_camera',
        'crossing1_34_thermal_camera',
        'crossing1_53_thermal_camera'
    ],
    'crossing2': [
        'crossing2_13_thermal_camera',
        'crossing2_14_thermal_camera',
        'crossing2_15_thermal_camera',
        'crossing2_33_thermal_camera',
        'crossing2_34_thermal_camera',
        'none'
    ],
    'crossing3': [
        'crossing3_13_thermal_camera',
        'crossing3_14_thermal_camera',
        'crossing3_15_thermal_camera',
        'crossing3_23_thermal_camera',
        'crossing3_24_thermal_camera',
        'crossing3_25_thermal_camera'
    ]
}

crossing_lidars = {
    'crossing1': [
        'crossing1_11_lidar',
        'crossing1_12_lidar',
        'crossing1_31_lidar',
        'crossing1_32_lidar'
    ],
    'crossing2': [
        'crossing2_11_lidar',
        'crossing2_12_lidar',
        'crossing2_31_lidar',
        'crossing2_32_lidar'
    ],
    'crossing3': [
        'crossing3_11_lidar',
        'crossing3_12_lidar',
        'crossing3_21_lidar',
        'crossing3_22_lidar'
    ]
}

vehicle_cameras = {
    'vehicle1': [
        'vehicle1_back_left_camera',
        'vehicle1_left_camera',
        'vehicle1_front_left_camera',
        'vehicle1_front_right_camera',
        'vehicle1_right_camera',
        'vehicle1_back_right_camera',
    ],
    'vehicle2': [
        'vehicle2_back_left_camera',
        'vehicle2_left_camera',
        'vehicle2_front_left_camera',
        'vehicle2_front_right_camera',
        'vehicle2_right_camera',
        'vehicle2_back_right_camera',
    ]
}

vehicle_lidars = {
    'vehicle1': [
        'vehicle1_middle_lidar',
    ],
    'vehicle2': [
        'vehicle2_middle_lidar',
    ]
}

vehicle_states = {
    'vehicle1': 'vehicle1_state',
    'vehicle2': 'vehicle2_state'
}

vehicle_colors = {  # bgr
    'vehicle1': (0, 0, 255),  # red
    'vehicle2': (255, 0, 0),  # blue
}

vehicle_states_id_mapping = {
    'vehicle1_state': 100000,
    'vehicle2_state': 200000,
    'crossing1': -1,
    'crossing2': -2,
    'crossing3': -3
}

vehicle1_lidar_index = [1]  # [1, 7, 13, 19]
outdoor_lidar_index = [2, 8, 14, 20]
crossing1_lidar_index = [3, 9, 15, 21]
crossing2_lidar_index = [4, 10, 16, 22]
crossing3_lidar_index = [5, 11, 17, 23]
vehicle2_lidar_index = [6]  # [6, 12, 18, 24]

lidar_index_mapping = {
    **{t: i for t, i in zip(crossing_lidars['crossing1'], crossing1_lidar_index)},
    **{t: i for t, i in zip(crossing_lidars['crossing2'], crossing2_lidar_index)},
    **{t: i for t, i in zip(crossing_lidars['crossing3'], crossing3_lidar_index)},
    **{t: i for t, i in zip(vehicle_lidars['vehicle1'], vehicle1_lidar_index)},
    **{t: i for t, i in zip(vehicle_lidars['vehicle2'], vehicle2_lidar_index)},
}

bl1, l1, fl1, fr1, r1, br1 = vehicle_cameras['vehicle1']
bl2, l2, fl2, fr2, r2, br2 = vehicle_cameras['vehicle2']
bl3, l3, fl3, fr3, r3, br3 = crossing_cameras['crossing1']
bl4, l4, fl4, fr4, r4, br4 = crossing_cameras['crossing2']
bl5, l5, fl5, fr5, r5, br5 = crossing_cameras['crossing3']
frame_level_pos_info = {
    **{bl1: (0, 0), l1: (1, 0), fl1: (2, 0), fr1: (3, 0), r1: (4, 0), br1: (5, 0)},
    **{bl2: (0, 1), l2: (1, 1), fl2: (2, 1), fr2: (3, 1), r2: (4, 1), br2: (5, 1)},
    **{bl3: (0, 2), l3: (1, 2)},
    **{fl3: (0, 3), fr3: (1, 3)},
    **{r3: (0, 4), br3: (1, 4)},
    **{bl4: (0, 2), l4: (1, 2)},
    **{fl4: (0, 3), fr4: (1, 3)},
    **{r4: (0, 4), br4: (1, 4)},
    **{bl5: (0, 2), l5: (1, 2)},
    **{fl5: (0, 3), fr5: (1, 3)},
    **{r5: (0, 4), br5: (1, 4)},
}

gps_origins = {
    'crossing1': (48.771731, 11.438043, 419),
    'crossing2': (48.772450, 11.441743, 419),
    'crossing3': (48.769060, 11.438518, 419),
}

# ground_params = {  # with only states
#     'crossing1': [-6.32662449e-05, -2.01889628e-05, -2.02631123e-05,  1.36400142e-03, 7.26723337e-05,  3.80466627e-01],
#     'crossing2': [ 5.22864204e-06,  5.61826591e-06, -4.59121710e-05, -4.05358545e-03, 1.54556150e-03,  4.26684155e-01],
#     'crossing3': [ 5.43821942e-05, -2.27764165e-05,  6.22164573e-05, -4.43234139e-03, 2.13904875e-03, -2.05336238e-01],
# }

ground_params = {  # states and labels
    'crossing1': [-2.17646045e-05, -2.54249987e-05, -8.18657797e-05,  3.81083323e-03, 8.94509838e-04,  4.57465017e-01],
    'crossing2': [-1.06222423e-05, -2.00344008e-05,  2.03389314e-05, -1.28649645e-03, -3.67867551e-04,  5.02631044e-01],
    'crossing3': [-1.21443378e-05, -1.32533272e-05,  3.58418801e-05,  1.78362570e-03, 2.67349162e-03, -7.98810367e-02],
}

# ground_params = { # states < 30m
#     'crossing1': [-0.0002519077294041521, -0.0001985239030466994, -4.2491469767478034e-05, -0.001394149018872422, 0.001348526777358048, 0.4975215230229767],
#     'crossing2': [-8.08576690589262e-05, -0.00027696961944360704, 0.0001751439465562411, -0.005629239959437079, 0.004883528296162676, 0.4775998681257498],
#     'crossing3': [-0.00038634689300494357, -0.00011367482201832447, 0.00012049498162828162, -0.009142698566119543, 0.002231060796454622, -0.11701059595772732]
# }