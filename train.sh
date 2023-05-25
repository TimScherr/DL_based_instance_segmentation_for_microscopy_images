eval "$(conda shell.bash hook)"

conda activate /srv/scherr/virtual_environments/kit-sch-ge-2021-cell_segmentation_ve

# ADAM
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "distance" --loss "smooth_l1" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "dist" --loss "smooth_l1" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "boundary" --loss "ce_dice" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "border" --loss "ce_dice" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "adapted_border" --loss "bce+ce" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "relu" --optimizer "adam" --method "j4" --loss "j_reg_loss" --iterations 11 --multi_gpu

# RANGER
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "distance" --loss "smooth_l1" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "dist" --loss "smooth_l1" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "boundary" --loss "ce_dice" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "border" --loss "ce_dice" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "adapted_border" --loss "bce+ce" --iterations 11 --multi_gpu
python ./train.py --cell_type "BF-C2DL-HSC" "BF-C2DL-MuSC" "Fluo-N2DL-HeLa" "PhC-C2DL-PSC" --mode "GT+ST" --split "01" --act_fun "mish" --optimizer "ranger" --method "j4" --loss "j_reg_loss" --iterations 11 --multi_gpu

conda deactivate

