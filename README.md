# Submission of IJCB-SynFacePAD-2023
The structure of our code is mainly developped based on the work [SynthASpoof](https://github.com/meilfang/SynthASpoof)[<sup>1</sup>](#refer-anchor-1).

## Environment setup
Some critical packages in our environment:
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

All the pre-prossing codes can be found in the `./data_preprocess` folder. You will need to change the path according to your environment.

For cropping facial regions, we applied the pre-prossing code provided by the organizers. The file structures are kept the same after the pre-processing.

We detected and saved a bonding box of the eye region using MTCNNN for each image, which is used for the eye cutout augmentation during training. 
```
python eye_crop.py
```

We also add jpg compression to the images after cropping. We saved 2 compressed versions of every image, keeping 50% and 25% of the original quality, respectively. Run the following line to compress the images. Notices that you will need to change the path in the code to cover every image.
```
python compress.py
```

We also use the number of each image as the corresponding id number. Run the following line to add this id information to the csv file. 
```
python get_id.py
```



## Training
Our method is a fusion of two models. Run the following two lines to train each model seperately:
```
sh train_pad.sh
sh train_id.sh
```
You will need to modify the settings in both `.sh` files according to you environment:

```
    --train_dir the root dir of training set
    --train_csv the csv file generated during pre-processing
    --test_dir the root dir of test set, set to None to disable testing
    --test_csv the coresponding test csv file of the test set
    --log name for wandb logging, set to None to disable wandb logging
```

The joint of `train_dir` and `image_path` in `train_csv` should form a complete path to the training images, and same goes for the test images.

The trained weights will be saved to `./pth` by defult, you may change this directory using `--pth_path=you_saving_dir`

Please refer to the code for other detailed parameter settings.

## Testing
The test set should also contain a corresponding csv file with `image_path` and `true_label`.

Run:
```
CUDA_VISIBLE_DEVICES=0 python test.py --test_data_dir=testset_root_dir --test_csv=testset_csv 
```
Where the joint of `testset_root_dir` and `image_path` in `testset_csv` should form a complete path to the test image.

Weights are loaded from `./pth` by defult, you may change this by setting `--model_path=your_model_dir`. Notice that this code will test all the models saved in the directory and output an averaged prediction score. Our pretrained weights can be found [here](https://drive.google.com/drive/folders/1wswcb8HW-OLI4IkptlqUlqjQQ82Z33N6?usp=share_link).

Results will be saved to `./result` by defult, you may change this by `--output_file=your_result_dir`.

The results will be saved in a `.csv` file, containing the following columns:`img_path`, `true_label`, `prediction_score`, `prediction_label`. The threshold when predicting lables is chosen to achieve the highest AUC (Area Under the Curve) value under given ground truth.

## Reference

<div id="refer-anchor-1"></div>
-[1] Meiling Fang, Marco Huber, and Naser Damer: SynthASpoof: Developing Face Presentation Attack Detection Based on Privacy-friendly Synthetic Data. 2023.
