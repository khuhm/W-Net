# W-Net: Two-Stage U-Net With Misaligned Data for Raw-to-RGB Mapping
## Environment
* Ubuntu 16.04 LTS.
* Pytorch 1.2.
* CUDA 10.0.

## Run inference on ZRR test images
1. Set argument 'raw_dir' to the directory containing input raw images

2. Download the pretrained models in [here](https://drive.google.com/drive/folders/1eH_prE7EWEUqxJes5IfHTdsFyqP64o7w?usp=sharing), and place them under the folder named "trained_model"

* To obtain the result images of 'fidelity track', run test.py

* To obtain the result images of 'perceptual track', run test_perceptual.py

* To obtain the full resolution result images, run test_fullres.py
