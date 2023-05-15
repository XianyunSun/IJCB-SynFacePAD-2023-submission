CUDA_VISIBLE_DEVICES=3 python train_id.py \
    --train_dir /sdata/xianyun.sun/SynthASpoof_data_crop \
    --train_csv /sdata/xianyun.sun/SynthASpoof_data_crop/labels.csv \
    --test_dir /sdata/xianyun.sun/casia_MFSD_img/test_release_crop  \
    --test_csv /sdata/xianyun.sun/casia_MFSD_img/labels.csv  \
    --train_epoch 1  \
    --log train_id