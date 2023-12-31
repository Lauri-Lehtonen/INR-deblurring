from __future__ import print_function, division

import argparse
import os
import sys
import pandas as pd
from fnmatch import fnmatchcase
from argparse import Namespace

from configs.utils import get_dataset_info, read_config
from modules.utils import read_log, search_with_pattern


def parser_fn(argv):
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config.yaml', help='Path to config file')
    parser.add_argument("--merge_ckpts", type=str, default=None, help='Merge ckpts from downloaded folder to results folder')
    args = parser.parse_args(argv)
    return args


def merge_ckpts(df, dataset, downloaded_folder, results_folder):
    """Merge ckpts from downloaded folder to results folder"""
    models = ['ICB', 'PWB']
    for model in models:
        for img_exp in df['img_exp'].values:
            src_file = os.path.join(downloaded_folder, 'deblur', dataset, model, 'ckpt', '{}.pth'.format(img_exp))
            dst_file = os.path.join(results_folder, 'deblur', dataset, model, 'ckpt', '{}.pth'.format(img_exp))
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            # move file
            os.system('mv {} {}'.format(src_file, dst_file))


def virtualCMB_info(dataset_info):
    """Generate info for VirtualCMB dataset"""
    # Get param logs
    param_logs = search_with_pattern(os.path.join(dataset_info.ROOT, 'param_logs'), '*txt')
    param_logs.sort()

    # Build dataframe
    d = {
        'img_exp': param_logs
    }
    df = pd.DataFrame(d)
    df['img_exp'] = df['img_exp'].map(lambda filename: os.path.basename(filename).replace('.txt', ''))

    df['scene'] = ''
    df['trans_mode'] = ''
    df['rot_mode'] = ''
    df['tag'] = ''

    tags = ['Macro', 'Trucking', 'Standard']
    for i, logfile in enumerate(param_logs):
        params = read_log(logfile)
        df.at[i, 'scene'] = os.path.basename(params['sceneName'])
        df.at[i, 'trans_mode'] = params['traslationMode']
        df.at[i, 'rot_mode'] = params['rotationMode']
        for tag in tags:
            if fnmatchcase(os.path.basename(df.loc[i]['img_exp']), '*{}_*'.format(tag)):
                df.at[i, 'tag'] = tag
                break
            else:
                continue
    
    return df


def realCMB_info(dataset_info):
    """Generate info for RealCMB dataset"""
    # Get blurry paths
    blurry_paths = search_with_pattern(os.path.join(dataset_info.ROOT, 'blurry'), '*png')
    blurry_paths.sort()

    # Build dataframe
    d = {
        'img_exp': blurry_paths
    }
    df = pd.DataFrame(d)
    df['img_exp'] = df['img_exp'].map(lambda filename: os.path.basename(filename).replace('.png', ''),)

    return df


info_functions={
    'VirtualCMB': virtualCMB_info,
    'RealCMB': realCMB_info
}


def main(argv=None): 
    args = parser_fn(argv)

    CONFIG = read_config(args.config)

    for dataset_info in CONFIG['DATASETS']:
        dataset_info = Namespace(**dataset_info)
        dataset = dataset_info.NAME
        info_func = info_functions[dataset]
        df = info_func(dataset_info)

        info_dir = os.path.dirname(dataset_info.INFO_CSV)
        os.makedirs(info_dir, exist_ok=True)

        df.to_csv(dataset_info.INFO_CSV, index=False)
        print('Dataframe successfully saved in: {} ({} images)'.format(dataset_info.INFO_CSV, len(df)))

        if args.merge_ckpts is not None:
            downloaded_folder = args.merge_ckpts
            results_folder = CONFIG['RESULTS_DIR']
            merge_ckpts(df, dataset, downloaded_folder, results_folder)
            print('Ckpts for {} successfully merged from {} to {}'.format(dataset, downloaded_folder, results_folder))

    if args.merge_ckpts is not None:
        # Remove recursive downloaded folder
        os.system('rm -rf {}'.format(downloaded_folder))

if __name__ == '__main__':
    main()
