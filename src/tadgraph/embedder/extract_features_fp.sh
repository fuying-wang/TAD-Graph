# CUDA_VISIBLE_DEVICES=5,6 python extract_features_fp.py --task tcga_read --data_h5_dir extracted_mag20x_patch256 --model uni
# CUDA_VISIBLE_DEVICES=5,6 python extract_features_fp.py --task tcga_ucec --data_h5_dir extracted_mag20x_patch256 --model uni
CUDA_VISIBLE_DEVICES=6,7 python extract_features_fp.py --task cptac_rcc --data_h5_dir extracted_mag20x_patch256 --model uni --batch_size 256
CUDA_VISIBLE_DEVICES=6,7 python extract_features_fp.py --task cptac_brca --data_h5_dir extracted_mag20x_patch256 --model uni --batch_size 256