#改进0：只读检查配置指定的KITTI训练/验证文件；不加载模型，不运行评估，不修改配置或数据。
import argparse
import pickle
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pcdet.utils.kitti_data_utils import (
    inspect_kitti_samples, format_kitti_data_report, rebuild_kitti_db_infos,
)


def merge_dataset_config(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dataset_config(base[key], value)
        else:
            base[key] = value
    return base


def load_dataset_config(cfg_file):
    with open(cfg_file, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    dataset = config.get('DATA_CONFIG', config)
    if '_BASE_CONFIG_' in dataset:
        base_path = Path(dataset['_BASE_CONFIG_'])
        if not base_path.is_absolute() and not base_path.is_file():
            base_path = PROJECT_ROOT / base_path
        base = load_dataset_config(base_path)
        dataset = merge_dataset_config(base, dataset)
    return dataset


def inspect_gt_database(root, dataset_cfg, class_names=None, max_examples=8):
    """Check the configured GT index and objects eligible for sampling, without edits."""
    root = Path(root)
    report = {'index_files': 0, 'objects': 0, 'missing_files': 0, 'examples': []}

    def record_missing(path, reason=''):
        report['missing_files'] += 1
        if len(report['examples']) < max_examples:
            report['examples'].append(str(path.resolve()) + ('; ' + reason if reason else ''))

    augmentor = dataset_cfg.get('DATA_AUGMENTOR', {})
    configs = augmentor if isinstance(augmentor, list) else augmentor.get('AUG_CONFIG_LIST', [])
    disabled = [] if isinstance(augmentor, list) else augmentor.get('DISABLE_AUG_LIST', [])
    for config in configs:
        if config['NAME'] != 'gt_sampling' or config['NAME'] in disabled:
            continue
        sample_groups = dict(group.split(':') for group in config.get('SAMPLE_GROUPS', []))
        selected_classes = list(sample_groups) if class_names is None else list(class_names)
        prepare = config.get('PREPARE', {})
        min_points = dict(item.split(':') for item in prepare.get('filter_by_min_points', []))
        excluded_difficulty = prepare.get('filter_by_difficulty', [])
        packed_paths = list(config.get('DB_DATA_PATH', []))
        for name in config['DB_INFO_PATH']:
            path = root / name
            report['index_files'] += 1
            db_infos = None
            if not path.is_file():
                backup = config.get('BACKUP_DB_INFO')
                if backup and len(config['DB_INFO_PATH']) == 1:
                    path = root / backup['DB_INFO_PATH']
                    packed_paths = [backup['DB_DATA_PATH']]
                elif path.name == 'kitti_dbinfos_train.pkl' \
                        and int(config.get('NUM_POINT_FEATURES', 0)) == 4:
                    try:
                        db_infos = rebuild_kitti_db_infos(
                            root, num_point_features=config.get('NUM_POINT_FEATURES', 4),
                        )
                    except (FileNotFoundError, ValueError) as error:
                        record_missing(path, 'same-dataset in-memory recovery failed: %s' % error)
                        continue
                if db_infos is None and not path.is_file():
                    record_missing(path, 'GT sampling index; restore the original index and object files')
                    continue
            if db_infos is None:
                with path.open('rb') as stream:
                    db_infos = pickle.load(stream)
            for class_name in selected_classes:
                if class_name not in db_infos:
                    record_missing(path, 'index has no entries for configured class %s' % class_name)
                    continue
                if int(sample_groups.get(class_name, 0)) <= 0:
                    continue
                for info in db_infos[class_name]:
                    if excluded_difficulty and info['difficulty'] in excluded_difficulty:
                        continue
                    minimum = int(min_points.get(class_name, 0))
                    if minimum > 0 and info['num_points_in_gt'] < minimum:
                        continue
                    report['objects'] += 1
                    if not config.get('USE_SHARED_MEMORY', False):
                        object_path = root / info['path']
                        if not object_path.is_file():
                            record_missing(object_path)
        if config.get('USE_SHARED_MEMORY', False):
            if not packed_paths:
                record_missing(root, 'USE_SHARED_MEMORY requires DB_DATA_PATH')
            for name in packed_paths:
                path = root / name
                if not path.is_file():
                    record_missing(path, 'packed GT database')
    return report


def main():
    parser = argparse.ArgumentParser(description='Read-only KITTI data check before training')
    parser.add_argument('--cfg_file', required=True, type=Path)
    parser.add_argument('--data_path', type=Path, default=None,
                        help='check this mount only; does not change the training configuration')
    args = parser.parse_args()
    dataset = load_dataset_config(args.cfg_file)
    root = args.data_path if args.data_path is not None else Path(dataset['DATA_PATH'])
    print('KITTI root: %s' % root.resolve())
    with args.cfg_file.open(encoding='utf-8') as stream:
        class_names = yaml.safe_load(stream).get('CLASS_NAMES')
    db_report = inspect_gt_database(root, dataset, class_names)
    print('GT database check: index_files={index_files}, sampled_objects={objects}, '
          'unavailable_files={missing_files}'.format(**db_report))
    for example in db_report['examples']:
        print(example)
    failed = bool(db_report['missing_files'])
    for mode in ('train', 'test'):
        split = dataset['DATA_SPLIT'][mode]
        infos = []
        for name in dataset['INFO_PATH'][mode]:
            path = root / name
            if not path.is_file():
                print('MISSING info file: %s' % path.resolve())
                failed = True
                continue
            with path.open('rb') as stream:
                infos.extend(pickle.load(stream))
        print('Mode=%s, split=%s' % (mode, split))
        if not infos:
            print('ERROR: no sample records; check INFO_PATH and the dataset mount.')
            failed = True
            continue
        report = inspect_kitti_samples(root, split, infos, dataset.get('GET_ITEM_LIST', ['points']))
        print(format_kitti_data_report(report))
        failed = failed or bool(report['missing_files'])
    if failed:
        print('FAIL: restore the reported files before rerunning the same training command.')
        return 1
    print('PASS: required sample files / same-sample calibration caches and GT sampling files '
          'are available. Point cloud and image contents were not checked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
#end
