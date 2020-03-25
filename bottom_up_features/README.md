# Bottom-Up Features Extractor

This code implements an extraction of Bottom-up image features ([paper](https://arxiv.org/abs/1707.07998)). Based on the original [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention/) and [PyTorch implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch).

## Requirements
* Python 3.6
* PyTorch 0.4.0
* CUDA 9.0

**Note:** CPU version is not supported.

## Installation
1. Install PyTorch with pip:
    ```
    pip install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
    ```
    or with Anaconda:
    ```
    conda install pytorch=0.4.0 cuda90 -c pytorch
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Compile the code:
    ```
    cd lib
    sh make.sh
    ```

4. Download the pretrained model from [dropbox](https://www.dropbox.com/s/qo4xf1dx3oxi1h6/bottomup_pretrained_10_100.pth?dl=0) or [google drive](https://drive.google.com/file/d/10MBUgH_OygyEys59FNQ4qNGeQ9bl-ODb/view?usp=drivesdk) and put it in models/ folder.

## Feature Extraction

1. To extract image features and store them in .npy format:
    ```
    python3 extract_features.py --image_dir ./raw_image_folder
                                --out_file_name name_of_out_file
                                --extraction_order_file ./fichier_alignement_video
                                --min_boxes nb_min_box
                                --max_boxes nb_max_box
                                [--boxes] (pour sauver les boundings boxes)
    ```

