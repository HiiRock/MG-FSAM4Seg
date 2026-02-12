<div align="center">
    <h2>
        Fine-tuned SAM adaptation with multi-scale guidance for automated detection toward image-based core length and RQD measurement
    </h2>
</div>

# Introduction
Rock core logging is critical for evaluating rock mass quality, yet traditional manual measurement of rock quality designation (RQD) indicator from boreholes remains highly labor-intensive. To address the limitations of insufficient detection accuracy and generalizability in mainstream deep learning methods, we propose MG-FSAM4Seg, a vision foundation model adapted for rock core instance segmentation, combined with a fine-grained RQD analytics method. MG-FSAM4Seg adopts an enhanced multi-scale feature learning design that seamlessly integrates the Segment Anything Model (SAM) encoder with additional decoders, where the encoder is further fine-tuned via low-rank adaptation. 

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


# Acknowledgements
> This project is developed based on [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/release) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main).  
> We sincerely thank the authors of these projects for their open-source contributions and support.

# License
This project is licensed under the [Apache 2.0 license](LICENSE).
