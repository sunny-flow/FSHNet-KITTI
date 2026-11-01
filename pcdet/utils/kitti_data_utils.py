#改进0：KITTI文件完整性检查与同帧标定缓存恢复；不修改样本、增强或训练参数。
from pathlib import Path

import numpy as np
import pickle


def rebuild_kitti_db_infos(root_path, info_filename='kitti_infos_train.pkl',
                           num_point_features=4, class_names=None):
    """Recreate the original KITTI GT-sampling index in memory from its source infos."""
    root_path = Path(root_path)
    info_path = root_path / info_filename
    if not info_path.is_file():
        raise FileNotFoundError('KITTI source info is missing: %s' % info_path.resolve())
    try:
        with info_path.open('rb') as stream:
            source_infos = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as error:
        raise ValueError('cannot read KITTI source info: %s' % info_path.resolve()) from error
    if not isinstance(source_infos, list) or not source_infos:
        raise ValueError('KITTI source info must be a non-empty list: %s' % info_path.resolve())

    split_path = root_path / 'ImageSets' / 'train.txt'
    if not split_path.is_file():
        raise FileNotFoundError('KITTI training split file is missing: %s' % split_path.resolve())
    try:
        split_text = split_path.read_text()
    except (OSError, UnicodeError) as error:
        raise ValueError('cannot read KITTI training split: %s' % split_path.resolve()) from error
    split_ids = [line.strip() for line in split_text.splitlines() if line.strip()]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError('KITTI training split contains duplicate sample IDs: %s' % split_path.resolve())

    try:
        feature_count = int(num_point_features)
    except (TypeError, ValueError) as error:
        raise ValueError('NUM_POINT_FEATURES must be a positive integer') from error
    if feature_count <= 0 or feature_count != num_point_features:
        raise ValueError('NUM_POINT_FEATURES must be a positive integer')
    row_bytes = feature_count * np.dtype(np.float32).itemsize

    db_infos = {}
    source_ids = []
    seen_ids = set()
    for frame in source_infos:
        try:
            sample_idx = frame['point_cloud']['lidar_idx']
            image_idx = frame['image']['image_idx']
            annos = frame['annos']
            names = annos['name']
            boxes = annos['gt_boxes_lidar']
            fov_point_counts = annos['num_points_in_gt']
            difficulties = annos['difficulty']
            bboxes = annos['bbox']
            scores = annos['score']
        except (KeyError, TypeError) as error:
            raise ValueError('incomplete KITTI source info in %s' % info_path.resolve()) from error
        sample_text = str(sample_idx)
        if sample_text != str(image_idx):
            raise ValueError('KITTI image/lidar sample ID mismatch: %s / %s' % (image_idx, sample_idx))
        if sample_text in seen_ids:
            raise ValueError('duplicate KITTI source sample ID: %s' % sample_text)
        if Path(sample_text).name != sample_text or '/' in sample_text or '\\' in sample_text:
            raise ValueError('invalid KITTI source sample ID: %s' % sample_text)
        seen_ids.add(sample_text)
        source_ids.append(sample_text)

        names_array = np.asarray(names)
        boxes_array = np.asarray(boxes)
        difficulties_array = np.asarray(difficulties)
        bboxes_array = np.asarray(bboxes)
        scores_array = np.asarray(scores)
        point_counts_array = np.asarray(fov_point_counts)
        try:
            boxes_are_finite = bool(np.isfinite(boxes_array).all())
        except TypeError:
            boxes_are_finite = False
        if boxes_array.ndim != 2 or boxes_array.shape[1] != 7 or not boxes_are_finite:
            raise ValueError('invalid KITTI gt_boxes_lidar for sample %s' % sample_text)
        num_objects = int(boxes_array.shape[0])
        if names_array.ndim != 1 or point_counts_array.ndim != 1 \
                or difficulties_array.ndim != 1 or scores_array.ndim != 1 \
                or bboxes_array.ndim != 2 or bboxes_array.shape[1] != 4:
            raise ValueError('invalid KITTI annotation shapes for sample %s' % sample_text)
        if any(len(values) < num_objects for values in
               (names_array, point_counts_array, difficulties_array, bboxes_array, scores_array)):
            raise ValueError('inconsistent KITTI annotation lengths for sample %s' % sample_idx)
        try:
            annotations_are_finite = all(np.isfinite(values[:num_objects]).all() for values in
                                         (point_counts_array, difficulties_array,
                                          bboxes_array, scores_array))
        except TypeError:
            annotations_are_finite = False
        if not annotations_are_finite:
            raise ValueError('non-finite KITTI annotation for sample %s' % sample_text)
        for gt_idx in range(num_objects):
            name = names[gt_idx]
            if class_names is not None and str(name) not in class_names:
                continue
            class_text = str(name)
            if not class_text or Path(class_text).name != class_text \
                    or '/' in class_text or '\\' in class_text:
                raise ValueError('invalid KITTI class name for sample %s object %d'
                                 % (sample_text, gt_idx))
            filename = '%s_%s_%d.bin' % (sample_idx, name, gt_idx)
            object_path = root_path / 'gt_database' / filename
            if not object_path.is_file():
                raise FileNotFoundError('KITTI GT object file is missing: %s' % object_path.resolve())
            file_bytes = object_path.stat().st_size
            if file_bytes % row_bytes:
                raise ValueError(
                    'invalid KITTI GT object byte size: %s (%d bytes, %d features)'
                    % (object_path.resolve(), file_bytes, feature_count)
                )
            point_count = file_bytes // row_bytes
            try:
                fov_point_count = int(point_counts_array[gt_idx])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('invalid FOV point count for sample %s object %d'
                                 % (sample_text, gt_idx)) from error
            if fov_point_count < 0 or point_count < fov_point_count:
                raise ValueError(
                    'invalid KITTI GT point counts for sample %s object %d: all=%d, fov=%d'
                    % (sample_text, gt_idx, point_count, fov_point_count)
                )
            record = {
                'name': name,
                'path': (Path('gt_database') / filename).as_posix(),
                'image_idx': sample_idx,
                'gt_idx': gt_idx,
                'box3d_lidar': boxes_array[gt_idx].copy(),
                'num_points_in_gt': point_count,
                'difficulty': difficulties[gt_idx],
                'bbox': bboxes_array[gt_idx].copy(),
                'score': scores[gt_idx],
            }
            db_infos.setdefault(name, []).append(record)
    if source_ids != split_ids:
        raise ValueError(
            'kitti_infos_train.pkl sample IDs/order do not match ImageSets/train.txt '
            '(%d source records versus %d split IDs)' % (len(source_ids), len(split_ids))
        )
    return db_infos


def calibration_from_info(info, sample_idx):
    """Recover the original float32 matrices saved by KittiDataset.get_infos."""
    if not isinstance(info, dict):
        raise ValueError('no matching KITTI info record is available')
    if str(info.get('point_cloud', {}).get('lidar_idx')) != str(sample_idx):
        raise ValueError('calibration cache belongs to a different lidar sample')
    image_idx = info.get('image', {}).get('image_idx', sample_idx)
    if str(image_idx) != str(sample_idx):
        raise ValueError('image and lidar sample IDs do not match')

    cached = info.get('calib')
    if not isinstance(cached, dict):
        raise ValueError('the matching info record has no calibration matrices')
    matrices = {}
    for source, target, shape in (
        ('P2', 'P2', (3, 4)),
        ('R0_rect', 'R0', (3, 3)),
        ('Tr_velo_to_cam', 'Tr_velo2cam', (3, 4)),
    ):
        if source not in cached:
            raise ValueError('missing cached calibration matrix: %s' % source)
        try:
            matrix = np.asarray(cached[source], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('invalid cached calibration matrix: %s' % source) from error
        if matrix.shape not in (shape, (4, 4)) or not np.isfinite(matrix).all():
            raise ValueError('invalid shape or non-finite cached matrix: %s' % source)
        matrices[target] = matrix[:shape[0], :shape[1]].copy()

    if matrices['P2'][0, 0] == 0 or matrices['P2'][1, 1] == 0:
        raise ValueError('cached calibration has zero focal length')
    for key in ('R0', 'Tr_velo2cam'):
        if abs(float(np.linalg.det(matrices[key][:3, :3]))) < 1e-8:
            raise ValueError('cached calibration has a singular rotation: %s' % key)
    return matrices


def inspect_kitti_samples(root_path, split, infos, get_item_list=('points',), max_examples=8):
    """Check exactly the configured samples; never substitute another split."""
    root_split_path = Path(root_path) / ('testing' if split == 'test' else 'training')
    report = {
        'samples': len(infos), 'calib_from_info': 0, 'missing_files': 0,
        'examples': [], 'root_split_path': str(root_split_path.resolve()),
    }

    def record_missing(path, reason=''):
        report['missing_files'] += 1
        if len(report['examples']) < max_examples:
            report['examples'].append(str(path.resolve()) + ('; ' + reason if reason else ''))

    for info in infos:
        sample_idx = info['point_cloud']['lidar_idx']
        calib_path = root_split_path / 'calib' / ('%s.txt' % sample_idx)
        if not calib_path.is_file():
            try:
                calibration_from_info(info, sample_idx)
            except ValueError as error:
                record_missing(calib_path, str(error))
            else:
                report['calib_from_info'] += 1
        for item, folder, suffix in (
            ('points', 'velodyne', '.bin'),
            ('images', 'image_2', '.png'),
            ('depth_maps', 'depth_2', '.png'),
        ):
            if item in get_item_list:
                path = root_split_path / folder / ('%s%s' % (sample_idx, suffix))
                if not path.is_file():
                    record_missing(path)
    return report


def format_kitti_data_report(report):
    message = (
        'KITTI data check: samples={samples}, calibration_from_info={calib_from_info}, '
        'unavailable_required_files={missing_files}; split_directory={root_split_path}'
    ).format(**report)
    if report['examples']:
        message += '\n' + '\n'.join(report['examples'])
    if report['missing_files']:
        message += (
            '\nRestore these files from the same KITTI split or correct the dataset mount. '
            'KITTI train and val both use training/, not testing/. '
            'No samples have been skipped and no calibration has been fabricated.'
        )
    return message
#end
