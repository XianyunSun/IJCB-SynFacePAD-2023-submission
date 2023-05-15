CUDA_VISIBLE_DEVICES=3 python train_pad.py \
    --train_dir /sdata/xianyun.sun/SynthASpoof_data_crop \
    --train_csv /sdata/xianyun.sun/SynthASpoof_data_crop/labels.csv \
    --test_dir /sdata/xianyun.sun/casia_MFSD_img/test_release_crop  \
    --test_csv /sdata/xianyun.sun/casia_MFSD_img/labels.csv  \
    --fine_tune_epoch 29  \
    --log train_pad