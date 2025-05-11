import os
import random


def ecs_final_split(opencood_dataset_dir, scenario_names, target_dir):
    crossing_1_scenarios = []
    crossing_2_scenarios = []
    crossing_3_scenarios = []

    for scenario_name in scenario_names:
        _split = scenario_name.split('_')
        crossing_name = _split[-2]
        if crossing_name == 'crossing1':
            crossing_1_scenarios.append(scenario_name)
        elif crossing_name == 'crossing2':
            crossing_2_scenarios.append(scenario_name)
        elif crossing_name == 'crossing3':
            crossing_3_scenarios.append(scenario_name)
        else:
            raise ValueError(f'Unknown crossing name: {crossing_name}')

    #################################################################
    ###################### EQUAL CROSSING SPLIT #####################
    #################################################################

    # WE CREATE 3 RANDOM SPLITS WITH RANDOM TRAIN, VAL, TEST SPLITS
    # crossing 1 [17, 2, 3] crossing_1_scenarios
    # crossing 2 [10, 2, 2] crossing_2_scenarios
    # crossing 3 [7, 2, 2] crossing_3_scenarios

    random.shuffle(crossing_1_scenarios)
    random.shuffle(crossing_2_scenarios)
    random.shuffle(crossing_3_scenarios)

    # Create crossing 1 splits
    c1_used_for_test = set()
    c1_used_for_val = set()
    available_for_val_test = set(crossing_1_scenarios)
    crossing_1_random_splits_train = []
    crossing_1_random_splits_val = []
    crossing_1_random_splits_test = []
    for i in range(3):
        remaining_for_val = sorted(available_for_val_test - c1_used_for_val - c1_used_for_test)
        val = random.sample(remaining_for_val, 2)
        c1_used_for_val.update(val)

        remaining_for_test = sorted(available_for_val_test - c1_used_for_val - c1_used_for_test)
        test = random.sample(remaining_for_test, 3)
        c1_used_for_test.update(test)

        train = [s for s in crossing_1_scenarios if s not in val and s not in test]

        crossing_1_random_splits_val.append(val)
        crossing_1_random_splits_test.append(test)
        crossing_1_random_splits_train.append(train)


    # Create crossing 2 splits
    c2_used_for_test = set()
    c2_used_for_val = set()
    available_for_val_test = set(crossing_2_scenarios)
    crossing_2_random_splits_train = []
    crossing_2_random_splits_val = []
    crossing_2_random_splits_test = []
    for i in range(3):
        remaining_for_val = sorted(available_for_val_test - c2_used_for_val)
        val = random.sample(remaining_for_val, 2)
        c2_used_for_val.update(val)

        remaining_for_test = sorted(available_for_val_test - c2_used_for_test)
        test = random.sample(remaining_for_test, 2)
        c2_used_for_test.update(test)

        train = [s for s in crossing_2_scenarios if s not in val and s not in test]

        crossing_2_random_splits_val.append(val)
        crossing_2_random_splits_test.append(test)
        crossing_2_random_splits_train.append(train)

    # Create crossing 3 splits
    c3_used_for_test = set()
    c3_used_for_val = set()
    available_for_val_test = set(crossing_3_scenarios)
    crossing_3_random_splits_train = []
    crossing_3_random_splits_val = []
    crossing_3_random_splits_test = []
    for i in range(3):
        remaining_for_val = sorted(available_for_val_test - c3_used_for_val)
        val = random.sample(remaining_for_val, 2)
        c3_used_for_val.update(val)

        remaining_for_test = sorted(available_for_val_test - c3_used_for_test)
        test = random.sample(remaining_for_test, 2)
        c3_used_for_test.update(test)

        train = [s for s in crossing_3_scenarios if s not in val and s not in test]

        crossing_3_random_splits_val.append(val)
        crossing_3_random_splits_test.append(test)
        crossing_3_random_splits_train.append(train)
    
    # create training splits
    for i in range(3):
        c1_train, c2_train, c3_train = crossing_1_random_splits_train[i], crossing_2_random_splits_train[i], crossing_3_random_splits_train[i]
        c1_val, c2_val, c3_val = crossing_1_random_splits_val[i], crossing_2_random_splits_val[i], crossing_3_random_splits_val[i]
        c1_test, c2_test, c3_test = crossing_1_random_splits_test[i], crossing_2_random_splits_test[i], crossing_3_random_splits_test[i]
        # flatten train
        train_flat = c1_train + c2_train + c3_train
        val_flat = c1_val + c2_val + c3_val
        test_flat = c1_test + c2_test + c3_test

        ecs_train_target_dir_split = os.path.join(target_dir, str(i), 'train')
        ecs_val_target_dir_split = os.path.join(target_dir, str(i), 'val')
        ecs_test_target_dir_split = os.path.join(target_dir, str(i), 'test')

        print(i, train_flat)
        print(i, val_flat)
        print(i, test_flat)

        os.makedirs(ecs_train_target_dir_split, exist_ok=True)
        os.makedirs(ecs_val_target_dir_split, exist_ok=True)
        os.makedirs(ecs_test_target_dir_split, exist_ok=True)

        # create symlinks
        for train_scenario in train_flat:
            os.symlink(os.path.join(opencood_dataset_dir, train_scenario), os.path.join(ecs_train_target_dir_split, train_scenario))
        for val_scenario in val_flat:
            os.symlink(os.path.join(opencood_dataset_dir, val_scenario), os.path.join(ecs_val_target_dir_split, val_scenario))
        for test_scenario in test_flat:
            os.symlink(os.path.join(opencood_dataset_dir, test_scenario), os.path.join(ecs_test_target_dir_split, test_scenario))


def scs_final_split(opencood_dataset_dir, scenario_names, target_dir):
    crossing_1_scenarios = []
    crossing_2_scenarios = []
    crossing_3_scenarios = []

    for scenario_name in scenario_names:
        _split = scenario_name.split('_')
        crossing_name = _split[-2]
        if crossing_name == 'crossing1':
            crossing_1_scenarios.append(scenario_name)
        elif crossing_name == 'crossing2':
            crossing_2_scenarios.append(scenario_name)
        elif crossing_name == 'crossing3':
            crossing_3_scenarios.append(scenario_name)
        else:
            raise ValueError(f'Unknown crossing name: {crossing_name}')
    
    separate_crossing_splits = [
        (['crossing1'], ['crossing2', 'crossing3']),
        (['crossing2'], ['crossing1', 'crossing3']),
        (['crossing3'], ['crossing1', 'crossing2']),
        (['crossing1','crossing2'], ['crossing3']),
        (['crossing1','crossing3'], ['crossing2']),
        (['crossing2','crossing3'], ['crossing1'])
    ]

    # equal crossing splits
    # only one combination since we use the same train/val/test splits for comparability

    # crossing 1 training set, crossing 1 validation set, crossing 1 test set
    c1_all_classes_to_use = [c1 for c1 in crossing_1_scenarios if c1.endswith('00')]
    random.shuffle(c1_all_classes_to_use)
    c2_all_classes_to_use = [c2 for c2 in crossing_2_scenarios if c2.endswith('00')]
    random.shuffle(c2_all_classes_to_use)
    c3_all_classes_to_use = [c3 for c3 in crossing_3_scenarios if c3.endswith('00')]
    random.shuffle(c3_all_classes_to_use)

    # 2 val/test for c1, 2 val/test for c2, 1 val/test for c3
    c1_val = c1_all_classes_to_use[:4]
    c1_train = set(crossing_1_scenarios)
    for c1s in c1_val:
        c1_train.remove(c1s)
    c2_val = c2_all_classes_to_use[:3]
    c2_train = set(crossing_2_scenarios)
    for c2s in c2_val:
        c2_train.remove(c2s)
    c3_val = c3_all_classes_to_use[:3]
    c3_train = set(crossing_3_scenarios)
    for c3s in c3_val:
        c3_train.remove(c3s)

    ###############################################################
    ################## SEPARATE CROSSING SPLITS ###################
    ###############################################################
    # create symlinks
    # fill dictionary with crossing names and scenarios
    for tup in separate_crossing_splits:
        train_val_seqs = tup[0]
        test_seqs = tup[1]
        
        split_name = f'scs_train_{"_".join(train_val_seqs)}_test_{"_".join(test_seqs)}'
        train_target_dir_split = os.path.join(target_dir, split_name, 'train')
        val_target_dir_split = os.path.join(target_dir, split_name, 'val')
        test_target_dir_split = os.path.join(target_dir, split_name, 'test')
        os.makedirs(train_target_dir_split, exist_ok=True)
        os.makedirs(val_target_dir_split, exist_ok=True)
        os.makedirs(test_target_dir_split, exist_ok=True)

        # create symlinks for train set
        for crossing in train_val_seqs:
            if crossing == 'crossing1':
                for scenario in c1_train:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(train_target_dir_split, scenario))
            elif crossing == 'crossing2':
                for scenario in c2_train:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(train_target_dir_split, scenario))
            elif crossing == 'crossing3':
                for scenario in c3_train:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(train_target_dir_split, scenario))

        # create symlinks for val set
        for crossing in train_val_seqs:
            if crossing == 'crossing1':
                for scenario in c1_val:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(val_target_dir_split, scenario))
            elif crossing == 'crossing2':
                for scenario in c2_val:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(val_target_dir_split, scenario))
            elif crossing == 'crossing3':
                for scenario in c3_val:
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(val_target_dir_split, scenario))
        
        # create symlinks for test set
        for crossing in test_seqs:
            if crossing == 'crossing1':
                for scenario in list([*c1_train, *c1_val]):
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(test_target_dir_split, scenario))
            elif crossing == 'crossing2':
                for scenario in list([*c2_train, *c2_val]):
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(test_target_dir_split, scenario))
            elif crossing == 'crossing3':
                for scenario in list([*c3_train, *c3_val]):
                    os.symlink(os.path.join(opencood_dataset_dir, scenario), os.path.join(test_target_dir_split, scenario))


if __name__ == '__main__':
    random.seed(42)
    split_type = 'scs'  # scs or ecs

    opencood_dataset_dir = r'/path/to/opencood/dataset'

    target_dir = os.path.join(opencood_dataset_dir, 'splits', split_type)
    train_target_dir = os.path.join(target_dir, 'train')
    val_target_dir = os.path.join(target_dir, 'val')
    test_target_dir = os.path.join(target_dir, 'test')

    # read scenario names
    _scenario_names = sorted(os.listdir(opencood_dataset_dir))
    scenario_names = []
    for scenario_name in _scenario_names:
        _split = scenario_name.split('_')
        if len(_split) < 3:
            # this is not a crossing
            continue
        elif not _split[-2].startswith('crossing'):
            # this is not a crossing
            continue
        else:
            scenario_names.append(scenario_name)

    if split_type == 'scs':
        scs_final_split(opencood_dataset_dir, scenario_names, target_dir)
    elif split_type == 'ecs':
        ecs_final_split(opencood_dataset_dir, scenario_names, target_dir)
    