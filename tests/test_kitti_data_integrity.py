#改进0：独立验证KITTI同帧标定恢复、文件预检和真实Dataset调用链，不依赖CUDA或外部数据集。
import ast
import copy
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA_UTILS = load_module('isolated_kitti_data_utils', 'pcdet/utils/kitti_data_utils.py')
CALIBRATION = load_module('isolated_calibration_kitti', 'pcdet/utils/calibration_kitti.py')


def make_info(sample_idx='000123', homogeneous=True):
    matrices = {
        'P2': np.array([
            [700.0, 0.0, 600.0, -35.0],
            [0.0, 710.0, 180.0, 1.5],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        'R0_rect': np.array([
            [np.cos(0.01), -np.sin(0.01), 0.0],
            [np.sin(0.01), np.cos(0.01), 0.0],
            [0.0, 0.0, 1.0],
        ]),
        'Tr_velo_to_cam': np.array([
            [0.0, -1.0, 0.0, 0.2],
            [0.0, 0.0, -1.0, 0.1],
            [1.0, 0.0, 0.0, -0.3],
        ]),
    }
    if homogeneous:
        for key, matrix in matrices.items():
            expanded = np.eye(4)
            expanded[:matrix.shape[0], :matrix.shape[1]] = matrix
            matrices[key] = expanded
    return {
        'point_cloud': {'lidar_idx': sample_idx, 'num_features': 4},
        'image': {'image_idx': sample_idx, 'image_shape': np.array([375, 1242])},
        'calib': matrices,
    }


def write_calibration(path, info):
    cached = info['calib']
    projection = cached['P2'][:3, :4]
    entries = [
        ('P0', projection), ('P1', projection), ('P2', projection), ('P3', projection),
        ('R0_rect', cached['R0_rect'][:3, :3]),
        ('Tr_velo_to_cam', cached['Tr_velo_to_cam'][:3, :4]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(
        name + ': ' + ' '.join(format(float(value), '.12g') for value in matrix.flat) + '\n'
        for name, matrix in entries
    ), encoding='utf-8')


def write_sample_files(root, split, info, items=('points',), calibration=False):
    sample_idx = info['point_cloud']['lidar_idx']
    base = root / ('testing' if split == 'test' else 'training')
    for item, folder, suffix in (
        ('points', 'velodyne', '.bin'),
        ('images', 'image_2', '.png'),
        ('depth_maps', 'depth_2', '.png'),
    ):
        if item in items:
            path = base / folder / (str(sample_idx) + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            if item == 'points':
                path.write_bytes(np.array([
                    [12.0, -1.0, 0.2, 0.5], [24.0, 2.0, -0.5, 0.9],
                ], dtype=np.float32).tobytes())
            else:
                path.touch()
    if calibration:
        write_calibration(base / 'calib' / (str(sample_idx) + '.txt'), info)


class FixtureTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='kitti-integrity-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class TestCalibrationFromInfo(FixtureTestCase):
    def test_accepts_saved_and_original_matrix_shapes(self):
        reference = make_info(homogeneous=False)['calib']
        for homogeneous in (False, True):
            with self.subTest(homogeneous=homogeneous):
                recovered = DATA_UTILS.calibration_from_info(make_info(homogeneous=homogeneous), '000123')
                self.assertEqual(set(recovered), {'P2', 'R0', 'Tr_velo2cam'})
                for source, target in (
                    ('P2', 'P2'), ('R0_rect', 'R0'), ('Tr_velo_to_cam', 'Tr_velo2cam'),
                ):
                    self.assertEqual(recovered[target].dtype, np.float32)
                    np.testing.assert_array_equal(recovered[target], reference[source].astype(np.float32))

    def test_ids_are_compared_as_strings_and_image_id_is_optional(self):
        info = make_info(sample_idx=123)
        recovered = DATA_UTILS.calibration_from_info(info, '123')
        self.assertEqual(recovered['P2'].shape, (3, 4))
        info['image'].pop('image_idx')
        DATA_UTILS.calibration_from_info(info, '123')
        info.pop('image')
        DATA_UTILS.calibration_from_info(info, '123')

    def test_rejects_wrong_lidar_or_image_frame(self):
        with self.assertRaises(ValueError):
            DATA_UTILS.calibration_from_info(make_info(), '000124')
        info = make_info()
        info['image']['image_idx'] = '000124'
        with self.assertRaises(ValueError):
            DATA_UTILS.calibration_from_info(info, '000123')

    def test_rejects_missing_required_fields(self):
        invalid_infos = [None, {}, {'point_cloud': {'lidar_idx': '000123'}}]
        for key in ('P2', 'R0_rect', 'Tr_velo_to_cam'):
            info = make_info()
            del info['calib'][key]
            invalid_infos.append(info)
        info = make_info()
        del info['point_cloud']['lidar_idx']
        invalid_infos.append(info)
        for index, info in enumerate(invalid_infos):
            with self.subTest(case=index), self.assertRaises(ValueError):
                DATA_UTILS.calibration_from_info(info, '000123')

    def test_rejects_invalid_shapes_and_nonnumeric_values(self):
        for key in ('P2', 'R0_rect', 'Tr_velo_to_cam'):
            for invalid in (np.eye(2), np.zeros((4, 3)), np.zeros(12), [['invalid']], None):
                with self.subTest(key=key, shape=np.shape(invalid)):
                    info = make_info()
                    info['calib'][key] = invalid
                    with self.assertRaises(ValueError):
                        DATA_UTILS.calibration_from_info(info, '000123')

    def test_rejects_nonfinite_values_including_homogeneous_padding(self):
        for key in ('P2', 'R0_rect', 'Tr_velo_to_cam'):
            for value in (np.nan, np.inf, -np.inf):
                for position in ((0, 0), (3, 3)):
                    with self.subTest(key=key, value=value, position=position):
                        info = make_info()
                        info['calib'][key][position] = value
                        with self.assertRaises(ValueError):
                            DATA_UTILS.calibration_from_info(info, '000123')

    def test_rejects_singular_rotations(self):
        for key in ('R0_rect', 'Tr_velo_to_cam'):
            with self.subTest(key=key):
                info = make_info()
                info['calib'][key][1, :3] = info['calib'][key][0, :3]
                with self.assertRaises(ValueError):
                    DATA_UTILS.calibration_from_info(info, '000123')

    def test_rejects_zero_focal_lengths(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                info = make_info()
                info['calib']['P2'][axis, axis] = 0
                with self.assertRaises(ValueError):
                    DATA_UTILS.calibration_from_info(info, '000123')

    def test_returns_copies_without_mutating_cached_matrices(self):
        info = make_info()
        original = copy.deepcopy(info)
        recovered = DATA_UTILS.calibration_from_info(info, '000123')
        for matrix in recovered.values():
            matrix[:] = 0
        for key in original['calib']:
            np.testing.assert_array_equal(info['calib'][key], original['calib'][key])

    def test_real_calibration_matches_text_projection_and_inverse(self):
        points = np.array([[10.0, -1.0, 0.3], [25.0, 2.0, -0.5]], dtype=np.float32)
        for homogeneous in (False, True):
            with self.subTest(homogeneous=homogeneous):
                info = make_info(homogeneous=homogeneous)
                text_path = self.root / '000123.txt'
                write_calibration(text_path, info)
                from_text = CALIBRATION.Calibration(text_path)
                from_info = CALIBRATION.Calibration(DATA_UTILS.calibration_from_info(info, '000123'))
                text_pixels, text_depth = from_text.lidar_to_img(points)
                info_pixels, info_depth = from_info.lidar_to_img(points)
                self.assertTrue(np.isfinite(info_pixels).all())
                np.testing.assert_allclose(info_pixels, text_pixels, rtol=1e-6, atol=1e-5)
                np.testing.assert_allclose(info_depth, text_depth, rtol=1e-6, atol=1e-5)
                np.testing.assert_allclose(
                    from_info.rect_to_lidar(from_info.lidar_to_rect(points)), points, atol=1e-5,
                )


class TestInspectKittiSamples(FixtureTestCase):
    def test_complete_samples_use_expected_split_directory(self):
        for split in ('train', 'val', 'test'):
            with self.subTest(split=split):
                info = make_info()
                write_sample_files(self.root, split, info, calibration=True)
                report = DATA_UTILS.inspect_kitti_samples(self.root, split, [info])
                directory = self.root / ('testing' if split == 'test' else 'training')
                self.assertEqual(report, {
                    'samples': 1, 'calib_from_info': 0, 'missing_files': 0,
                    'examples': [], 'root_split_path': str(directory.resolve()),
                })

    def test_existing_text_does_not_validate_cache(self):
        info = make_info()
        write_sample_files(self.root, 'train', info, calibration=True)
        info['calib'] = {'P2': None}
        info['image']['image_idx'] = 'different-frame'
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'train', [info])
        self.assertEqual(report['missing_files'], 0)
        self.assertEqual(report['calib_from_info'], 0)

    def test_valid_cache_recovers_only_calibration_and_preserves_infos(self):
        infos = [make_info('000321'), make_info('005678')]
        original = copy.deepcopy(infos)
        for info in infos:
            write_sample_files(self.root, 'train', info)
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'val', infos)
        self.assertEqual(report['samples'], 2)
        self.assertEqual(report['calib_from_info'], 2)
        self.assertEqual(report['missing_files'], 0)
        self.assertEqual(len(infos), len(original))
        for before, after in zip(original, infos):
            self.assertEqual(before['point_cloud'], after['point_cloud'])
            for key in before['calib']:
                np.testing.assert_array_equal(before['calib'][key], after['calib'][key])

    def test_checks_only_requested_payload_files(self):
        info = make_info()
        write_sample_files(self.root, 'train', info)
        default = DATA_UTILS.inspect_kitti_samples(self.root, 'train', [info])
        self.assertEqual(default['missing_files'], 0)
        report = DATA_UTILS.inspect_kitti_samples(
            self.root, 'train', [info], ('points', 'images', 'depth_maps'),
        )
        self.assertEqual(report['missing_files'], 2)
        self.assertTrue(any('image_2' in path and '000123.png' in path for path in report['examples']))
        self.assertTrue(any('depth_2' in path and '000123.png' in path for path in report['examples']))
        write_sample_files(self.root, 'train', info, items=('images', 'depth_maps'))
        self.assertEqual(DATA_UTILS.inspect_kitti_samples(
            self.root, 'train', [info], ('points', 'images', 'depth_maps'),
        )['missing_files'], 0)

    def test_no_points_requested_does_not_require_velodyne(self):
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'train', [make_info()], ())
        self.assertEqual(report['missing_files'], 0)
        self.assertEqual(report['calib_from_info'], 1)

    def test_never_uses_testing_files_for_train_or_val(self):
        info = make_info()
        write_sample_files(self.root, 'test', info, calibration=True)
        del info['calib']
        for split in ('train', 'val'):
            with self.subTest(split=split):
                report = DATA_UTILS.inspect_kitti_samples(self.root, split, [info])
                self.assertEqual(report['samples'], 1)
                self.assertEqual(report['missing_files'], 2)
                self.assertEqual(report['calib_from_info'], 0)
                self.assertTrue(all('training' in path for path in report['examples']))

    def test_does_not_borrow_adjacent_info_or_skip_invalid_sample(self):
        infos = [make_info('000123'), make_info('000124')]
        del infos[1]['calib']
        for info in infos:
            write_sample_files(self.root, 'train', info)
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'train', infos)
        self.assertEqual(report['samples'], 2)
        self.assertEqual(report['calib_from_info'], 1)
        self.assertEqual(report['missing_files'], 1)
        self.assertIn('000124.txt', report['examples'][0])

    def test_cache_for_wrong_image_is_reported_missing(self):
        info = make_info()
        write_sample_files(self.root, 'train', info)
        info['image']['image_idx'] = '000124'
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'train', [info])
        self.assertEqual(report['calib_from_info'], 0)
        self.assertEqual(report['missing_files'], 1)
        self.assertIn('000123.txt', report['examples'][0])

    def test_missing_files_count_is_not_truncated_with_examples(self):
        infos = [make_info('000123'), make_info('000124'), make_info('000125')]
        for limit in (0, 2):
            with self.subTest(limit=limit):
                report = DATA_UTILS.inspect_kitti_samples(
                    self.root, 'train', infos, ('points', 'images', 'depth_maps'), max_examples=limit,
                )
                self.assertEqual(report['samples'], 3)
                self.assertEqual(report['calib_from_info'], 3)
                self.assertEqual(report['missing_files'], 9)
                self.assertEqual(len(report['examples']), limit)

    def test_empty_infos_return_empty_report(self):
        report = DATA_UTILS.inspect_kitti_samples(self.root, 'train', [])
        self.assertEqual(report['samples'], 0)
        self.assertEqual(report['calib_from_info'], 0)
        self.assertEqual(report['missing_files'], 0)
        self.assertEqual(report['examples'], [])


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class DatasetTemplateStub:
    """Replace GPU preprocessing only; exercise the actual KITTI dataset methods."""
    def __init__(self, dataset_cfg, class_names, training, root_path, logger):
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.mode = 'train' if training else 'test'
        self.root_path = Path(root_path)
        self.logger = logger
        self._merge_all_iters_to_one_epoch = False

    def prepare_data(self, data_dict):
        return data_dict


def load_kitti_dataset_class():
    path = ROOT / 'pcdet/datasets/kitti/kitti_dataset.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    dataset_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'KittiDataset')
    isolated_ast = ast.Module(body=[dataset_class], type_ignores=[])
    namespace = {
        '__name__': 'isolated_kitti_dataset', 'DatasetTemplate': DatasetTemplateStub,
        'copy': copy, 'pickle': pickle, 'np': np, 'Path': Path,
        'calibration_kitti': CALIBRATION,
        'calibration_from_info': DATA_UTILS.calibration_from_info,
        'inspect_kitti_samples': DATA_UTILS.inspect_kitti_samples,
        'format_kitti_data_report': DATA_UTILS.format_kitti_data_report,
    }
    exec(compile(isolated_ast, str(path), 'exec'), namespace)
    return namespace['KittiDataset']


class TestKittiDatasetIntegration(FixtureTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_class = load_kitti_dataset_class()

    def make_dataset(self, infos, fov_only=False):
        info_path = self.root / 'kitti_infos_fixture.pkl'
        info_path.write_bytes(pickle.dumps(infos))
        cfg = Config(
            DATASET='KittiDataset',
            DATA_SPLIT={'train': 'train', 'test': 'val'},
            INFO_PATH={'train': [info_path.name], 'test': [info_path.name]},
            GET_ITEM_LIST=['points'], FOV_POINTS_ONLY=fov_only,
        )
        before = copy.deepcopy(cfg)
        dataset = self.dataset_class(cfg, ['Car', 'Pedestrian', 'Cyclist'], root_path=self.root)
        self.assertEqual(cfg, before)
        return dataset

    def test_getitem_uses_each_current_info_when_text_is_missing(self):
        infos = [make_info('000123'), make_info('000987')]
        infos[1]['calib']['P2'][0, 2] += 20.0
        for info in infos:
            write_sample_files(self.root, 'train', info)
        dataset = self.make_dataset(infos, fov_only=True)
        dataset.validate_required_files()
        self.assertEqual(len(dataset), 2)
        for index, info in enumerate(infos):
            result = dataset[index]
            self.assertEqual(result['frame_id'], info['point_cloud']['lidar_idx'])
            self.assertEqual(result['points'].shape, (2, 4))
            np.testing.assert_array_equal(result['image_shape'], info['image']['image_shape'])
            np.testing.assert_array_equal(result['calib'].P2, info['calib']['P2'][:3, :4].astype(np.float32))

    def test_getitem_text_wins_over_invalid_cached_calibration(self):
        info = make_info()
        write_sample_files(self.root, 'train', info, calibration=True)
        original_projection = info['calib']['P2'][:3, :4].astype(np.float32).copy()
        info['calib'] = None
        dataset = self.make_dataset([info])
        dataset.validate_required_files()
        np.testing.assert_array_equal(dataset[0]['calib'].P2, original_projection)
        np.testing.assert_array_equal(dataset.get_calib('000123').P2, original_projection)

    def test_get_calib_rejects_other_frame_and_names_missing_path(self):
        info = make_info()
        write_sample_files(self.root, 'train', info)
        dataset = self.make_dataset([info])
        with self.assertRaises(FileNotFoundError) as caught:
            dataset.get_calib('000999', info=info)
        self.assertIn('000999.txt', str(caught.exception))
        self.assertIn('training', str(caught.exception))

    def test_get_calib_does_not_borrow_available_info_when_not_supplied(self):
        info = make_info()
        write_sample_files(self.root, 'train', info)
        dataset = self.make_dataset([info])
        with self.assertRaises(FileNotFoundError):
            dataset.get_calib('000123')

    def test_explicit_precheck_rejects_missing_points_and_allows_offline_init(self):
        info = make_info()
        dataset = self.make_dataset([info])
        self.assertEqual(len(dataset), 1)
        with self.assertRaises(FileNotFoundError) as caught:
            dataset.validate_required_files()
        self.assertIn('velodyne', str(caught.exception))
        self.assertIn('000123.bin', str(caught.exception))

    def test_precheck_does_not_substitute_testing_data(self):
        info = make_info()
        write_sample_files(self.root, 'test', info, calibration=True)
        dataset = self.make_dataset([info])
        with self.assertRaises(FileNotFoundError) as caught:
            dataset.validate_required_files()
        self.assertIn(str(self.root / 'training'), str(caught.exception))
        self.assertIn('000123.bin', str(caught.exception))

    def test_build_dataloader_checks_files_before_constructing_loader(self):
        dataset = self.make_dataset([make_info()])
        path = ROOT / 'pcdet/datasets/__init__.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'build_dataloader')
        loader = Mock()
        namespace = {
            '__all__': {'KittiDataset': self.dataset_class},
            'KittiDataset': self.dataset_class, 'DataLoader': loader,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        with self.assertRaises(FileNotFoundError) as caught:
            namespace['build_dataloader'](
                dataset.dataset_cfg, ['Car'], batch_size=1, dist=False, root_path=self.root,
            )
        self.assertIn('000123.bin', str(caught.exception))
        loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
#end
