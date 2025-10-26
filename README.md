<div align="center">
    <h2>
        Fine-tuned SAM adaptation with multi-scale guidance for automated detection and RQD analytics in rock core imagery
    </h2>
</div>

# Introduction
Rock core logging is critical for evaluating rock mass quality, yet traditional manual measurement of rock quality designation (RQD) indicator from boreholes remains highly labor-intensive. To address the limitations of insufficient detection accuracy and generalizability in mainstream deep learning methods, we propose **SAM4CoreSeg**, a vision foundation model adapted for rock core instance segmentation, combined with a fine-grained RQD analytics method. **SAM4CoreSeg** adopts an enhanced multi-scale feature learning design that seamlessly integrates the Segment Anything Model (SAM) encoder with additional decoders. 

<div align="center">
  <img src="figures/visualization.png" width="800"/>
</div>
<br>

# Dependencies
This project is based on the MMDetection framework and supports the following environment:
- OS: Linux or Windows
- Python ≥ 3.7 (recommended 3.10)
- PyTorch ≥ 2.0 (recommended 2.1)
- CUDA ≥ 11.7 (recommended 12.1)
- MMCV ≥ 2.0 (recommended 2.1)

We recommend creating a conda virtual environment and installing PyTorch and MMCV first.
For detailed installation steps, please refer to the [MMDetection documentation](https://github.com/open-mmlab/mmdetection/tree/main).

# Dataset Preparation
```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/Core
├── annotations
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images
    ├── train
    ├── val
    └── test
```
Our dataset follows the COCO format and contains annotated rock core images with instance masks. For more support methods, please refer to [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main).

# Usage
## Train
Run the following command to start training:
```shell
python tools/train.py configs/xxx.py  # xxx.py is the configuration file you want to use
```

## Evaluate
To validate the model and visualize the results:
```shell
python tools/test.py configs/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```

## Inference
To perform image prediction and save the results:
```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/rsprompter/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE is the image file you want to predict, xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```

# Acknowledgements
> This project is developed based on [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/release) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main).  
> We sincerely thank the authors of these projects for their open-source contributions and support.

# License
This project is licensed under the [Apache 2.0 license](LICENSE).
