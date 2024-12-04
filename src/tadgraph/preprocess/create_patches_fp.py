# internal imports
from tadgraph.wsi_core.WholeSlideImage import WholeSlideImage
from tadgraph.wsi_core.wsi_utils import StitchCoords
from tadgraph.wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import ipdb
import pandas as pd
from tadgraph.paths import *


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(
        0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object: WholeSlideImage, seg_params: dict = None, filter_params: dict = None, mask_file: str = None):
    # Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    # Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object: WholeSlideImage, **kwargs):
    # Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    # Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size=256, step_size=256,
                  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                              'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level=0,
                  use_default_params=False,
                  seg=False, save_mask=True,
                  stitch=False,
                  patch=False, auto_skip=True, process_list=None):

    # TODO: need to change code for EBRAINS
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(
        os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params,
                           vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params,
                           vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        try:
            WSI_object = WholeSlideImage(full_path)
        except Exception as e:
            print('failed to open slide with error: {}'.format(e))
            df.loc[idx, 'status'] = 'failed_open'
            continue

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(
                        old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(
                str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(
                str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print(
                'level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            try:
                WSI_object, seg_time_elapsed = segment(
                    WSI_object, current_seg_params, current_filter_params)
            except Exception as e:
                print('segmentation failed with error: {}'.format(e))
                df.loc[idx, 'status'] = 'failed_seg'
                continue

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                                                     'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(
                WSI_object=WSI_object,  **current_patch_params,)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(
                    file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--task', type=str, default='camelyon16', 
                    choices=['camelyon16', 'tcga_nsclc', 'tcga_rcc', 'tcga_brca', 'tcga_esca', 
                             'tcga_blca', 'tcga_ucec', 'tcga_gbmlgg', 'tcga_prad',
                             'tcga_read', 'panda', 'ebrains', 'tcga_coad',
                             'cptac_rcc', 'cptac_brca'])
# parser.add_argument('--step_size', type=int, default=256,
#                     help='step_size')
parser.add_argument('--patch_size', type=int, default=512,
                    help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--patch_level', type=int, default=1,
                    help='downsample level at which to patch')
parser.add_argument('--process_list',  type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')
parser.add_argument('--objective', type=int, default=40,
                    help='Magnitude of WSI')

'''
python create_patches_fp.py --task tcga_prad --seg --patch --stitch 
'''

if __name__ == '__main__':
    args = parser.parse_args()

    # Default magnitude is 40x
    # we want to xxx
    args.mag = args.objective // (2 ** args.patch_level)

    if args.task == "camelyon16":
        args.source = os.path.join(CAMELYON16_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(CAMELYON16_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "bwh_biopsy.csv")
    elif args.task == "tcga_nsclc":
        args.source = os.path.join(NSCLC_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(NSCLC_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_rcc":
        args.source = os.path.join(RCC_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(RCC_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_brca":
        args.source = os.path.join(BRCA_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(BRCA_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_esca":
        args.source = os.path.join(ESCA_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(ESCA_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_blca":
        args.source = os.path.join(BLCA_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(BLCA_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_ucec":
        args.source = os.path.join(UCEC_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(UCEC_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_gbmlgg":
        args.source = os.path.join(GBMLGG_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(GBMLGG_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_prad":
        args.source = os.path.join(PRAD_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(PRAD_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "tcga_read":
        args.source = os.path.join(READ_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(READ_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "panda":
        args.source = os.path.join(PANDA_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(PANDA_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "bwh_biopsy.csv")
    elif args.task == "ebrains":
        args.source = os.path.join(EBRAINS_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(EBRAINS_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "bwh_biopsy.csv")
    elif args.task == "tcga_coad":
        args.source = os.path.join(COAD_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(COAD_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "cptac_rcc":
        args.source = os.path.join(CPTAC_RCC_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(CPTAC_RCC_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    elif args.task == "cptac_brca":
        args.source = os.path.join(CPTAC_BRCA_DATA_DIR, "WSIs")
        args.save_dir = os.path.join(CPTAC_BRCA_DATA_DIR, f"extracted_mag{args.mag}x_patch{args.patch_size}")
        args.preset = os.path.join(PRESET_DIR, "tcga.csv")
    else:
        raise NotImplementedError()
    args.step_size = args.patch_size

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None
        
    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size=args.patch_size, step_size=args.step_size,
                                           seg=args.seg,  use_default_params=False, save_mask=True,
                                           stitch=args.stitch,
                                           patch_level=args.patch_level, patch=args.patch,
                                           process_list=process_list, auto_skip=args.no_auto_skip)
