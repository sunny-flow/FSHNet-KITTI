#改进0：隔离真实GT采样器和只读检查函数，验证缺失文件、原备用分支与采样过滤，不依赖CUDA扩展。
import ast
import copy
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_definition(relative_path, name, extra_globals=None):
    path = ROOT / relative_path
    parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    definition = next(node for node in parsed.body
                      if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    namespace = {'__name__': 'isolated_gt_database', 'copy': copy, 'os': os,
                 'pickle': pickle, 'np': np, 'Path': Path}
    namespace.update(extra_globals or {})
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def sampler_config(**updates):
    config = Config(NAME='gt_sampling', DB_INFO_PATH=['kitti_dbinfos_train.pkl'],
                    DB_DATA_PATH=['gt_database.npy'], NUM_POINT_FEATURES=4,
                    PREPARE={}, SAMPLE_GROUPS=['Car:2'], USE_SHARED_MEMORY=False)
    config.update(updates)
    return config


def object_info(name='000001_Car_0.bin', difficulty=0, points=5):
    return {'path': 'gt_database/' + name, 'difficulty': difficulty,
            'num_points_in_gt': points, 'name': 'Car'}


def source_info(sample_idx='000001', names=('Car',), cached_points=0):
    count = len(names)
    return {
        'point_cloud': {'lidar_idx': sample_idx, 'num_features': 4},
        'image': {'image_idx': sample_idx},
        'annos': {
            'name': np.asarray(names),
            'gt_boxes_lidar': np.arange(count * 7, dtype=np.float64).reshape(count, 7) + 0.25,
            'difficulty': np.arange(count, dtype=np.int32),
            'bbox': np.arange(count * 4, dtype=np.float32).reshape(count, 4) + 0.5,
            'score': np.arange(count, dtype=np.float64) - 1.0,
            # get_infos counts FOV points, while GT object files contain all points.
            'num_points_in_gt': np.full(count, cached_points, dtype=np.int32),
        },
    }


class FixtureTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gt-db-integrity-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_index(self, infos, name='kitti_dbinfos_train.pkl'):
        path = self.root / name
        path.write_bytes(pickle.dumps(infos))
        return path

    def write_object(self, info):
        path = self.root / info['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.zeros((info['num_points_in_gt'], 4), dtype=np.float32).tobytes())
        return path

    def write_source_infos(self, infos, name='kitti_infos_train.pkl'):
        path = self.root / name
        path.write_bytes(pickle.dumps(infos))
        split_path = self.root / 'ImageSets' / 'train.txt'
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            ''.join('%s\n' % info['point_cloud']['lidar_idx'] for info in infos),
            encoding='utf-8',
        )
        return path

    def write_source_object(self, info, gt_idx, points, num_features=4):
        sample_idx = info['point_cloud']['lidar_idx']
        name = info['annos']['name'][gt_idx]
        path = self.root / 'gt_database' / ('%s_%s_%d.bin' % (sample_idx, name, gt_idx))
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.arange(points * num_features, dtype=np.float32).reshape(points, num_features)
        path.write_bytes(values.tobytes())
        return path


class TestRebuildKittiDbInfos(FixtureTestCase):
    @classmethod
    def setUpClass(cls):
        cls.rebuild = staticmethod(load_definition(
            'pcdet/utils/kitti_data_utils.py', 'rebuild_kitti_db_infos'))

    def test_rebuild_matches_original_fields_order_and_file_point_counts_read_only(self):
        frames = [source_info('000002', ('Car', 'Pedestrian')),
                  source_info('000001', ('Car',))]
        source_path = self.write_source_infos(frames)
        self.write_source_object(frames[0], 0, points=3)
        self.write_source_object(frames[0], 1, points=2)
        self.write_source_object(frames[1], 0, points=1)
        before = {path.relative_to(self.root).as_posix(): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}

        rebuilt = self.rebuild(self.root)

        after = {path.relative_to(self.root).as_posix(): path.read_bytes()
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(after, before)
        self.assertFalse((self.root / 'kitti_dbinfos_train.pkl').exists())
        self.assertEqual(source_path.read_bytes(), before['kitti_infos_train.pkl'])
        self.assertEqual(list(rebuilt), ['Car', 'Pedestrian'])
        self.assertEqual(
            [(record['image_idx'], record['gt_idx'], record['num_points_in_gt'])
             for record in rebuilt['Car']],
            [('000002', 0, 3), ('000001', 0, 1)],
        )
        self.assertEqual(
            [(record['image_idx'], record['gt_idx'], record['num_points_in_gt'])
             for record in rebuilt['Pedestrian']],
            [('000002', 1, 2)],
        )
        first = rebuilt['Car'][0]
        self.assertEqual(list(first), [
            'name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar',
            'num_points_in_gt', 'difficulty', 'bbox', 'score',
        ])
        self.assertEqual(first['name'], frames[0]['annos']['name'][0])
        self.assertEqual(first['path'], 'gt_database/000002_Car_0.bin')
        self.assertEqual(first['difficulty'], frames[0]['annos']['difficulty'][0])
        self.assertEqual(first['score'], frames[0]['annos']['score'][0])
        np.testing.assert_array_equal(first['box3d_lidar'], frames[0]['annos']['gt_boxes_lidar'][0])
        np.testing.assert_array_equal(first['bbox'], frames[0]['annos']['bbox'][0])
        self.assertNotEqual(first['num_points_in_gt'],
                            int(frames[0]['annos']['num_points_in_gt'][0]))

    def test_rebuild_reports_missing_source_and_missing_object(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.rebuild(self.root)
        self.assertIn(str((self.root / 'kitti_infos_train.pkl').resolve()), str(caught.exception))

        frame = source_info()
        self.write_source_infos([frame])
        with self.assertRaises(FileNotFoundError) as caught:
            self.rebuild(self.root)
        self.assertIn(str((self.root / 'gt_database' / '000001_Car_0.bin').resolve()),
                      str(caught.exception))

    def test_rebuild_rejects_object_bytes_that_do_not_form_complete_points(self):
        frame = source_info()
        self.write_source_infos([frame])
        path = self.root / 'gt_database' / '000001_Car_0.bin'
        path.parent.mkdir(parents=True)
        path.write_bytes(np.zeros(3, dtype=np.float32).tobytes())
        original = path.read_bytes()
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root, num_point_features=4)
        self.assertIn(str(path.resolve()), str(caught.exception))
        self.assertIn('12 bytes', str(caught.exception))
        self.assertEqual(path.read_bytes(), original)

    def test_rebuild_requires_training_split(self):
        frame = source_info()
        self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=1)
        split_path = self.root / 'ImageSets' / 'train.txt'
        split_path.unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            self.rebuild(self.root)
        self.assertIn(str(split_path.resolve()), str(caught.exception))

    def test_rebuild_rejects_duplicate_split_and_source_ids(self):
        frames = [source_info('000001'), source_info('000002')]
        self.write_source_infos(frames)
        split_path = self.root / 'ImageSets' / 'train.txt'
        split_path.write_text('000001\n000001\n', encoding='utf-8')
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        self.assertIn('duplicate sample IDs', str(caught.exception))

        frames[1]['point_cloud']['lidar_idx'] = '000001'
        frames[1]['image']['image_idx'] = '000001'
        self.write_source_infos(frames)
        split_path.write_text('000001\n000002\n', encoding='utf-8')
        self.write_source_object(frames[0], 0, points=1)
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        self.assertIn('duplicate KITTI source sample ID', str(caught.exception))

    def test_rebuild_rejects_image_lidar_frame_mismatch(self):
        frame = source_info('000123')
        frame['image']['image_idx'] = '000124'
        self.write_source_infos([frame])
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        self.assertIn('image/lidar sample ID mismatch', str(caught.exception))

    def test_rebuild_rejects_truncated_or_reordered_training_split(self):
        frames = [source_info('000001'), source_info('000002')]
        self.write_source_infos(frames)
        for frame in frames:
            self.write_source_object(frame, 0, points=1)
        split_path = self.root / 'ImageSets' / 'train.txt'
        for contents in ('000001\n', '000002\n000001\n'):
            with self.subTest(contents=contents):
                split_path.write_text(contents, encoding='utf-8')
                with self.assertRaises(ValueError) as caught:
                    self.rebuild(self.root)
                self.assertIn('sample IDs/order do not match', str(caught.exception))

    def test_rebuild_accepts_zero_point_object_when_fov_count_is_zero(self):
        frame = source_info(cached_points=0)
        self.write_source_infos([frame])
        object_path = self.write_source_object(frame, 0, points=0)
        self.assertEqual(object_path.stat().st_size, 0)
        rebuilt = self.rebuild(self.root)
        self.assertEqual(rebuilt['Car'][0]['num_points_in_gt'], 0)

    def test_rebuild_rejects_invalid_or_nonfinite_boxes(self):
        frame = source_info()
        self.write_source_infos([frame])
        frame['annos']['gt_boxes_lidar'] = np.zeros((1, 6), dtype=np.float64)
        self.write_source_infos([frame])
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        self.assertIn('invalid KITTI gt_boxes_lidar', str(caught.exception))

        frame['annos']['gt_boxes_lidar'] = np.zeros((1, 7), dtype=np.float64)
        frame['annos']['gt_boxes_lidar'][0, 3] = np.inf
        self.write_source_infos([frame])
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        self.assertIn('invalid KITTI gt_boxes_lidar', str(caught.exception))

    def test_rebuild_rejects_object_count_below_cached_fov_count(self):
        frame = source_info(cached_points=2)
        self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=1)
        with self.assertRaises(ValueError) as caught:
            self.rebuild(self.root)
        message = str(caught.exception)
        self.assertIn('all=1, fov=2', message)
        self.assertIn('000001', message)


class TestDataBaseSamplerGuard(FixtureTestCase):
    @classmethod
    def setUpClass(cls):
        cls.rebuild = staticmethod(load_definition(
            'pcdet/utils/kitti_data_utils.py', 'rebuild_kitti_db_infos'))
        cls.sampler_class = load_definition(
            'pcdet/datasets/augmentor/database_sampler.py', 'DataBaseSampler',
            {'rebuild_kitti_db_infos': cls.rebuild})

    def test_missing_index_and_source_names_both_paths_without_changing_config(self):
        for updates in ({}, {'BACKUP_DB_INFO': None}, {'BACKUP_DB_INFO': {}}):
            with self.subTest(updates=updates):
                config = sampler_config(**updates)
                before = copy.deepcopy(config)
                with self.assertRaises(FileNotFoundError) as caught:
                    self.sampler_class(self.root, config, ['Car'])
                message = str(caught.exception)
                for expected in (str((self.root / config.DB_INFO_PATH[0]).resolve()),
                                 str((self.root / 'kitti_infos_train.pkl').resolve()),
                                 'same-dataset KITTI recovery failed'):
                    self.assertIn(expected, message)
                self.assertEqual(config, before)

    def test_standard_kitti_index_is_rebuilt_in_memory_with_real_sampler(self):
        frame = source_info('000123', ('Car', 'Van'))
        source_path = self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=3)
        self.write_source_object(frame, 1, points=2)
        config = sampler_config(
            PREPARE={'filter_by_min_points': ['Car:2']}, SAMPLE_GROUPS=['Car:1'])
        before_config = copy.deepcopy(config)
        before_source = source_path.read_bytes()

        sampler = self.sampler_class(self.root, config, ['Car'])

        self.assertEqual(config, before_config)
        self.assertEqual(source_path.read_bytes(), before_source)
        self.assertFalse((self.root / 'kitti_dbinfos_train.pkl').exists())
        self.assertEqual(len(sampler.db_infos['Car']), 1)
        self.assertEqual(sampler.db_infos['Car'][0]['num_points_in_gt'], 3)
        self.assertEqual(sampler.db_infos['Car'][0]['path'], 'gt_database/000123_Car_0.bin')
        self.assertEqual(sampler.sample_class_num, {'Car': '1'})
        self.assertEqual(sampler.sample_groups['Car']['pointer'], 1)

    def test_missing_non_target_class_object_rejects_recovery(self):
        frame = source_info('000123', ('Car', 'Van'))
        self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=3)
        config = sampler_config(SAMPLE_GROUPS=['Car:1'])
        before = copy.deepcopy(config)
        with self.assertRaises(FileNotFoundError) as caught:
            self.sampler_class(self.root, config, ['Car'])
        self.assertIn('000123_Van_1.bin', str(caught.exception))
        self.assertEqual(config, before)

    def test_shared_memory_does_not_enable_in_memory_recovery(self):
        frame = source_info()
        self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=2)
        config = sampler_config(USE_SHARED_MEMORY=True)
        before = copy.deepcopy(config)
        original_destructor = self.sampler_class.__del__
        self.sampler_class.__del__ = lambda instance: None
        try:
            with self.assertRaises(FileNotFoundError) as caught:
                self.sampler_class(self.root, config, ['Car'])
        finally:
            self.sampler_class.__del__ = original_destructor
        self.assertIn('No compatible same-dataset recovery', str(caught.exception))
        self.assertEqual(config, before)
        self.assertFalse((self.root / 'kitti_dbinfos_train.pkl').exists())

    def test_missing_nonstandard_index_is_not_treated_as_kitti(self):
        frame = source_info()
        self.write_source_infos([frame])
        self.write_source_object(frame, 0, points=2)
        config = sampler_config(DB_INFO_PATH=['custom_dbinfos_train.pkl'])
        before = copy.deepcopy(config)
        with self.assertRaises(FileNotFoundError) as caught:
            self.sampler_class(self.root, config, ['Car'])
        self.assertIn(str((self.root / 'custom_dbinfos_train.pkl').resolve()),
                      str(caught.exception))
        self.assertIn('No compatible same-dataset recovery', str(caught.exception))
        self.assertEqual(config, before)
        self.assertFalse((self.root / 'kitti_dbinfos_train.pkl').exists())

    def test_existing_index_preserves_records_parameters_and_sample_count(self):
        infos = {'Car': [object_info(), object_info('000002_Car_0.bin')]}
        index = self.write_index(infos)
        for info in infos['Car']:
            self.write_object(info)
        original_bytes = index.read_bytes()
        config = sampler_config()
        before = copy.deepcopy(config)
        sampler = self.sampler_class(self.root, config, ['Car'])
        self.assertEqual(config, before)
        self.assertEqual(sampler.db_infos, infos)
        self.assertEqual(sampler.sample_class_num, {'Car': '2'})
        self.assertEqual(sampler.sample_groups['Car']['pointer'], 2)
        np.testing.assert_array_equal(sampler.sample_groups['Car']['indices'], [0, 1])
        self.assertEqual(len(sampler.sample_with_fixed_number('Car', sampler.sample_groups['Car'])), 2)
        self.assertEqual(index.read_bytes(), original_bytes)
        self.assertFalse(sampler.use_shared_memory)

    def test_waymo_backup_preserves_original_fallback_parameter_changes(self):
        infos = {'Car': [object_info()]}
        self.write_index(infos, 'original_backup.pkl')
        self.write_object(infos['Car'][0])
        config = sampler_config(DB_INFO_PATH=['waymo_primary_dbinfos.pkl'], BACKUP_DB_INFO={
            'DB_INFO_PATH': 'original_backup.pkl', 'DB_DATA_PATH': 'original_backup.npy',
            'NUM_POINT_FEATURES': 5,
        })
        sampler = self.sampler_class(self.root, config, ['Car'])
        self.assertEqual(sampler.db_infos, infos)
        self.assertEqual(config.DB_INFO_PATH, ['original_backup.pkl'])
        self.assertEqual(config.DB_DATA_PATH, ['original_backup.npy'])
        self.assertEqual(config.NUM_POINT_FEATURES, 5)


class TestInspectGtDatabase(FixtureTestCase):
    @classmethod
    def setUpClass(cls):
        rebuild = load_definition('pcdet/utils/kitti_data_utils.py', 'rebuild_kitti_db_infos')
        cls.inspect = staticmethod(load_definition(
            'tools/check_kitti_data.py', 'inspect_gt_database',
            {'rebuild_kitti_db_infos': rebuild}))

    def dataset_config(self, sampler=None, disabled=None):
        return {'DATA_PATH': str(self.root), 'DATA_AUGMENTOR': {
            'DISABLE_AUG_LIST': [] if disabled is None else disabled,
            'AUG_CONFIG_LIST': [sampler if sampler is not None else sampler_config()],
        }}

    def test_missing_index_is_counted_and_names_absolute_path(self):
        config = self.dataset_config()
        before = copy.deepcopy(config)
        report = self.inspect(self.root, config)
        self.assertEqual((report['index_files'], report['objects'], report['missing_files']), (1, 0, 1))
        self.assertIn(str((self.root / 'kitti_dbinfos_train.pkl').resolve()), report['examples'][0])
        self.assertEqual(config, before)

    def test_disabled_or_absent_gt_sampling_needs_no_files(self):
        for config in ({}, self.dataset_config(disabled=['gt_sampling']),
                       self.dataset_config(sampler_config(NAME='random_world_flip'))):
            with self.subTest(config=config):
                self.assertEqual(self.inspect(self.root, config), {
                    'index_files': 0, 'objects': 0, 'missing_files': 0, 'examples': [],
                })

    def test_filters_and_class_selection_only_check_eligible_objects(self):
        valid = object_info()
        missing = object_info('000002_Car_0.bin')
        infos = {'Car': [valid, missing, object_info('hard.bin', difficulty=-1),
                         object_info('sparse.bin', points=4)],
                 'Cyclist': [object_info('cyclist.bin')],
                 'Pedestrian': [object_info('pedestrian.bin')],
                 'Truck': [object_info('truck.bin')]}
        index = self.write_index(infos)
        self.write_object(valid)
        before_bytes = index.read_bytes()
        config = self.dataset_config(sampler_config(
            SAMPLE_GROUPS=['Car:2', 'Pedestrian:0', 'Cyclist:3'],
            PREPARE={'filter_by_difficulty': [-1], 'filter_by_min_points': ['Car:5']},
        ))
        before_config = copy.deepcopy(config)
        report = self.inspect(self.root, config, class_names=['Car', 'Pedestrian', 'Truck'])
        self.assertEqual((report['index_files'], report['objects'], report['missing_files']), (1, 2, 1))
        self.assertEqual(report['examples'], [str((self.root / missing['path']).resolve())])
        self.assertEqual(config, before_config)
        self.assertEqual(index.read_bytes(), before_bytes)

    def test_complete_objects_and_legacy_augmentor_list_pass(self):
        info = object_info()
        self.write_index({'Car': [info]})
        self.write_object(info)
        report = self.inspect(self.root, {'DATA_AUGMENTOR': [sampler_config()]})
        self.assertEqual(report, {'index_files': 1, 'objects': 1, 'missing_files': 0, 'examples': []})

    def test_directory_is_not_an_object_file(self):
        info = object_info()
        self.write_index({'Car': [info]})
        (self.root / info['path']).mkdir(parents=True)
        self.assertEqual(self.inspect(self.root, self.dataset_config())['missing_files'], 1)

    def test_example_limit_does_not_limit_missing_count(self):
        self.write_index({'Car': [object_info('missing_%d.bin' % i) for i in range(4)]})
        for limit in (0, 2):
            with self.subTest(limit=limit):
                report = self.inspect(self.root, self.dataset_config(), max_examples=limit)
                self.assertEqual(report['objects'], 4)
                self.assertEqual(report['missing_files'], 4)
                self.assertEqual(len(report['examples']), limit)

    def test_explicit_backup_is_checked_without_mutating_configuration(self):
        info = object_info()
        backup_index = self.write_index({'Car': [info]}, 'original_backup.pkl')
        self.write_object(info)
        config = self.dataset_config(sampler_config(BACKUP_DB_INFO={
            'DB_INFO_PATH': backup_index.name, 'DB_DATA_PATH': 'original_backup.npy',
            'NUM_POINT_FEATURES': 5,
        }))
        before = copy.deepcopy(config)
        original_bytes = backup_index.read_bytes()
        report = self.inspect(self.root, config)
        self.assertEqual(report, {'index_files': 1, 'objects': 1, 'missing_files': 0, 'examples': []})
        self.assertEqual(config, before)
        self.assertEqual(backup_index.read_bytes(), original_bytes)


if __name__ == '__main__':
    unittest.main()
#end
