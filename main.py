import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc

from datasets.TimeDataset import TimeDataset

from models.GDN import GDN
from models.GDN_UQ import GDN_UQ

from train import train
from test import test
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores

from datetime import datetime

import os
import json
import argparse
from pathlib import Path

import random


class Main():
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset, train_config['seed'], train_config['batch'],
            val_ratio=train_config['val_ratio'],
            split_path=env_config.get('split_path', ''),
            slide_win=train_config['slide_win'],
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0
        )

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        model_kwargs = dict(
            dim=train_config['dim'],
            input_dim=train_config['slide_win'],
            out_layer_num=train_config['out_layer_num'],
            out_layer_inter_dim=train_config['out_layer_inter_dim'],
            topk=train_config['topk'],
        )
        if env_config['model'] == 'gdn_uq':
            uq_kwargs = dict(model_kwargs)
            if 'logvar_clamp' in train_config:
                uq_kwargs['logvar_clamp'] = train_config['logvar_clamp']
            self.model = GDN_UQ(
                edge_index_sets, len(feature_map), **uq_kwargs
            ).to(self.device)
        else:
            self.model = GDN(
                edge_index_sets, len(feature_map), **model_kwargs
            ).to(self.device)

    def run(self):
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            print(f'CHECKPOINT_PATH={model_save_path}', flush=True)
            # Persist the actual training config alongside the checkpoint so
            # build_manifest.py can capture any non-default flags (e.g. the
            # tighter logvar_clamp and the logvar_l2 reg). RESULTS.md fix.
            import json as _json
            hp_path = Path(os.path.dirname(model_save_path)) / 'hyperparameters.json'
            _json.dump(
                {**self.train_config,
                 'logvar_clamp': list(self.train_config.get('logvar_clamp', [-10.0, 10.0])),
                 'model': self.env_config['model'],
                 'dataset': self.env_config['dataset']},
                hp_path.open('w'), indent=2,
            )

            self.train_log = train(
                self.model, model_save_path,
                config=self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )

        # Detection scoring is out of scope for the gdn_uq path; evaluate.py
        # assumes a single-tensor model output and would break on (mu, log_var).
        if self.env_config['model'] == 'gdn_uq':
            return

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1,
                    split_path='', slide_win=5):
        """Build train and val DataLoaders.

        If `split_path` is given (a JSON with `train_rows` and `val_rows`
        row-index tuples — same schema as `train_gdeltauq_main.py` consumes),
        use a deterministic split that matches the G-DeltaUQ pipeline. Row
        ranges are converted to window indices via the standard
        `max(0, r - slide_win)` mapping. The optional `aleatoric_rows`
        slice in the JSON is intentionally ignored here — GDN doesn't have
        an aleatoric head, so we just hold those rows out of training to
        match the G-DeltaUQ pipeline's effective train+val data.

        Otherwise fall back to the original random `val_start_index` split.
        """
        if split_path:
            with open(split_path) as f:
                split = json.load(f)
            total_windows = int(len(train_dataset))

            def _r2w(rr):
                return (max(0, rr[0] - slide_win),
                        min(total_windows, rr[1] - slide_win))

            tw0, tw1 = _r2w(split['train_rows'])
            vw0, vw1 = _r2w(split['val_rows'])
            train_subset = Subset(train_dataset, list(range(tw0, tw1)))
            val_subset = Subset(train_dataset, list(range(vw0, vw1)))
            print(f'split_path: {split_path}')
            print(f'  train windows: {len(train_subset)} '
                  f'({tw0}..{tw1})')
            print(f'  val windows:   {len(val_subset)} '
                  f'({vw0}..{vw1})')
        else:
            dataset_len = int(len(train_dataset))
            train_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = int(dataset_len * val_ratio)
            val_start_index = random.randrange(train_use_len)
            indices = torch.arange(dataset_len)

            train_sub_indices = torch.cat([
                indices[:val_start_index],
                indices[val_start_index + val_use_len:]
            ])
            train_subset = Subset(train_dataset, train_sub_indices)
            val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
            val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')

        if self.env_config.get('save_arrays', False):
            np_val_result = np.array(val_result)
            arrays_path = self.get_save_path()[1].replace('.csv', '_arrays.npz')
            np.savez(
                arrays_path,
                full_scores=test_scores.astype(np.float32),
                test_attack_label=np.array(test_labels, dtype=np.int8),
                test_predict=np_test_result[0].astype(np.float32),
                test_ground_truth=np_test_result[1].astype(np.float32),
                val_predict=np_val_result[0].astype(np.float32),
                val_ground_truth=np_val_result[1].astype(np.float32),
                F1=float(info[0]), precision=float(info[1]),
                recall=float(info[2]),
            )
            print(f'arrays saved to {arrays_path}')

    def get_save_path(self, feature_name=''):
        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-epoch', help='train epoch', type=int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='swat')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-topk', help='topk num', type=int, default=20)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-model', help='gdn / gdn_uq', type=str, default='gdn',
                        choices=['gdn', 'gdn_uq'])
    parser.add_argument('-logvar_clamp_low', type=float, default=-10.0,
                        help='lower bound on log_var clamp for GDN_UQ; the '
                             'paper-aligned tighter setting is -3.0.')
    parser.add_argument('-logvar_clamp_high', type=float, default=10.0,
                        help='upper bound on log_var clamp for GDN_UQ; the '
                             'paper-aligned tighter setting is 3.0.')
    parser.add_argument('-logvar_l2', type=float, default=0.0,
                        help='L2 penalty on log_var (β·mean(log_var²)) added to '
                             'Gaussian NLL; 0.01 is a sensible non-zero value.')
    parser.add_argument('-split_path', type=str, default='',
                        help='Optional path to a JSON split file with '
                             'train_rows/val_rows tuples (same schema as '
                             'train_gdeltauq_main.py uses). When given, '
                             'overrides val_ratio and uses a deterministic '
                             'split that matches the G-DeltaUQ pipeline. '
                             'Optional aleatoric_rows in the JSON are '
                             'ignored by GDN.')
    parser.add_argument('-save_arrays', action='store_true',
                        help='If set, persist arrays.npz alongside the results '
                             'CSV (full_scores, test_attack_label, test_predict, '
                             'test_ground_truth, val_predict, val_ground_truth) '
                             'so downstream threshold/post-proc experiments can '
                             'operate on the same artifacts as the GDeltaUQ '
                             'pipeline.')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'logvar_clamp': (args.logvar_clamp_low, args.logvar_clamp_high),
        'logvar_l2': args.logvar_l2,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'model': args.model,
        'save_arrays': args.save_arrays,
        'split_path': args.split_path,
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
