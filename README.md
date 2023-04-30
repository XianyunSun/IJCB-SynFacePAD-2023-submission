# Submission of IJCB-SynFacePAD-2023
The structure of our code is mainly developped based on the work [SynthASpoof](https://github.com/meilfang/SynthASpoof)[<sup>1</sup>](#refer-anchor-1).

## Environment setup
Some critial packages in our environment:
```
python==3.9.16
albumentations==1.3.0
facenet-pytorch==2.5.2
scikit-learn==1.0.2
timm==0.6.12
torch==1.13.1
torchvision==0.14.1
wandb==0.14.0
```

## Data pre-prossing
We applied the pre-prossing code provided by the organizers. The file structures are kept the same after the pre-processing.

## Training
To reproduce our method, run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --train_dir=dataset_root_dir --train_csv=dataset_csv --fine_tune_epoch=29 --log=train
```

Where `dataset_csv` is the csv file generated during pre-processing. The joint of `dataset_root_dir` and `image_path` in `dataset_csv` should form a complete path to the training image.

Seting `--log` to any non-empty string will enable wandb logging. Set this to `None` to disable.

The trained weights will be saved to `./pth` by defult, you may change this directory using `--pth_path=you_saving_dir`

Please refer to the code for other detailed parameter settings.

## Testing
The test set should also contain a corresponding csv file with `image_path` and `true_label`.

Run:
```
CUDA_VISIBLE_DEVICES=0 python test.py --test_data_dir=testset_root_dir --test_csv=testset_csv 
```
Where the joint of `testset_root_dir` and `image_path` in `testset_csv` should form a complete path to the test image.

Weights are loaded from `./pth` by defult, you may change this by setting `--model_path=your_mndel_dir`. Notice that this code will test all the models saved in the directory and output an averaged prediction score. Our pretrained weights can be found [here]([https://drive.google.com/file/d/13GqcEk2Jp2N7nBYhlheTAtUJAXzzKmeO/view?usp=share_link](https://drive.google.com/drive/folders/1wswcb8HW-OLI4IkptlqUlqjQQ82Z33N6?usp=share_link)).

Results will be saved to `./result` by defult, you may change this by `--output_file=your_result_dir`.

The results will be saved in a `.csv` file, containing the following columns:`img_path`, `true_label`, `prediction_score`, `prediction_label`. The threshold when predicting lables is chosen to achieve the highest AUC(Area Under the Curve) value under given ground truth.

## Reference

<div id="refer-anchor-1"></div>
-[1] Meiling Fang, Marco Huber, and Naser Damer: SynthASpoof: Developing Face Presentation Attack Detection Based on Privacy-friendly Synthetic Data. 2023.
