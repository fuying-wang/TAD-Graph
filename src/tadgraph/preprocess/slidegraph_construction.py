import argparse
import os
import h5py
import ipdb
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.transforms import Polar
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
from torch_geometric.transforms import LargestConnectedComponents
from tadgraph.paths import *


def toTensor(v, dtype=torch.float, requires_grad=True):
    device = 'cpu'
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)


def connectClusters(Cc, dthresh=3000):
    tess = Delaunay(Cc)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx = neighbors
    W = np.zeros((Cc.shape[0], Cc.shape[0]))
    for n in nx:
        nx[n] = np.array(list(nx[n]), dtype=np.int64)
        nx[n] = nx[n][KDTree(Cc[nx[n], :]).query_ball_point(Cc[n], r=dthresh)]
        W[n, nx[n]] = 1.0
        W[nx[n], n] = 1.0
    return W  # neighbors of each cluster and an affinity matrix


def toGeometric(X, W, y, tt=0):
    return Data(x=toTensor(X, requires_grad=False), edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(), y=toTensor([y], dtype=torch.long, requires_grad=False))


def pt2graph(wsi_h5, patch_size=1024, lambda_f=1e-3, lamda_h=0.8, distance_thres=4000):
    lambda_d = 1 / float(patch_size)
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]

    # ridx = (np.max(features, axis=0) - np.min(features, axis=0)
    #         ) > 1e-4  # remove feature which does not change
    # features = features[:, ridx]

    TC = sKDTree(coords)
    I, D = TC.query_radius(coords, r=6 / lambda_d,
                           return_distance=True, sort_results=True)

    DX = np.zeros(int(coords.shape[0] * (coords.shape[0] - 1) / 2))
    idx = 0
    for i in range(coords.shape[0] - 1):
        f = np.exp(-lambda_f *
                   np.linalg.norm(features[i] - features[I[i]], axis=1))
        d = np.exp(-lambda_d * D[i])
        df = 1 - f * d
        dfi = np.ones(coords.shape[0])
        dfi[I[i]] = df
        dfi = dfi[i + 1:]
        DX[idx:idx + len(dfi)] = dfi
        idx = idx + len(dfi)
    d = DX

    Z = hierarchy.linkage(d, method='average')
    clusters = fcluster(Z, lamda_h, criterion='distance')
    uc = list(set(clusters))
    C_cluster = []
    F_cluster = []
    for c in uc:
        idx = np.where(clusters == c)
        if coords[idx, :].squeeze().size == 2:
            C_cluster.append(list(np.round(coords[idx, :].squeeze())))
            F_cluster.append(list(features[idx, :].squeeze()))
        else:
            C_cluster.append(
                list(np.round(coords[idx, :].squeeze().mean(axis=0))))
            F_cluster.append(list(features[idx, :].squeeze().mean(axis=0)))
    C_cluster = np.array(C_cluster)
    F_cluster = np.array((F_cluster))

    W = connectClusters(C_cluster, dthresh=distance_thres)
    G = toGeometric(F_cluster, W, y=1)
    G.pos = toTensor(C_cluster, requires_grad=False)

    # # keep largest components
    # transform = LargestConnectedComponents(num_components=1)
    # G = transform(G)

    polar_transform = Polar()
    G = polar_transform(G)

    return G


def createDir_h5toPyG(h5_path, save_path, skip_existed=True, patch_size=1024):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        graph_file = os.path.join(save_path, h5_fname[:-3]+'.pt')

        if skip_existed:
            if os.path.exists(graph_file):
                continue

        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))

        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5, patch_size=patch_size)
            torch.save(G, graph_file)
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')


def main():
    '''
    python slidegraph_construction.py --dataset tcga_esca --skip_existed --patch_size 256
    python slidegraph_construction.py --dataset tcga_blca --skip_existed --patch_size 256
    python slidegraph_construction.py --dataset tcga_brca --skip_existed --patch_size 256
    '''
    parser = argparse.ArgumentParser(description="Construct WSI graph")
    parser.add_argument("--dataset", type=str, default="tcga_nsclc",
                        choices=["camelyon16", "tcga_nsclc", "tcga_rcc", "tcga_brca",
                                 "tcga_blca", "tcga_ucec", "tcga_esca", "tcga_prad"])
    parser.add_argument("--magnitude", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--skip_existed", action="store_false")
    parser.add_argument("--save_dir", type=str, default="slide_graph")
    args = parser.parse_args()

    # args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/vits_tcga_pancancer_dino_pt_patch_features/h5_files"
    # args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/vits_tcga_pancancer_dino_pt_patch_features"

    # args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/kimianet_pt_patch_features/h5_files"
    # args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/kimianet_pt_patch_features"

    args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/resnet50_trunc_pt_patch_features/h5_files"
    args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/resnet50_trunc_pt_patch_features"

    # args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/ctrans_pt_patch_features/h5_files"
    # args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/ctrans_pt_patch_features"

    # args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/uni_pt_patch_features/h5_files"
    # args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/uni_pt_patch_features"

    if args.dataset.lower() == "camelyon16":
        args.h5_path = os.path.join(CAMELYON16_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            CAMELYON16_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_nsclc":
        args.h5_path = os.path.join(NSCLC_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            NSCLC_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_rcc":
        args.h5_path = os.path.join(RCC_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            RCC_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_brca":
        args.h5_path = os.path.join(BRCA_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            BRCA_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_blca":
        args.h5_path = os.path.join(BLCA_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            BLCA_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_ucec":
        args.h5_path = os.path.join(UCEC_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            UCEC_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_esca":
        args.h5_path = os.path.join(ESCA_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            ESCA_DATA_DIR, args.save_path, args.save_dir)
    elif args.dataset.lower() == "tcga_prad":
        args.h5_path = os.path.join(PRAD_DATA_DIR, args.h5_path)
        args.save_path = os.path.join(
            PRAD_DATA_DIR, args.save_path, args.save_dir)
    else:
        raise RuntimeError(f"No dataset {args.dataset}!")

    os.makedirs(args.save_path, exist_ok=True)

    patch_size = args.patch_size * (2 ** (40 / args.magnitude - 1))
    createDir_h5toPyG(args.h5_path, args.save_path,
                      args.skip_existed, patch_size)


if __name__ == "__main__":
    main()
