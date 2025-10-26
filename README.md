<div align="center">
    <h2>
        Fine-tuned SAM adaptation with multi-scale guidance for automated detection and RQD analytics in rock core imagery
    </h2>
</div>

# Introduce
Rock core logging is critical for evaluating rock mass quality, yet traditional manual measurement of rock quality designation (RQD) indicator from boreholes remains highly labor-intensive. To address the limitations of insufficient detection accuracy and generalizability in mainstream deep learning methods, we propose SAM4CoreSeg, a vision foundation model adapted for rock core instance segmentation, combined with a fine-grained RQD analytics method. SAM4CoreSeg adopts an enhanced multi-scale feature learning design that seamlessly integrates the Segment Anything Model (SAM) encoder with additional decoders. 
<div align="center">
  <img src="Figures/Visualization of Test Set Prediction Results.jpg" width="800"/>
</div>
<br>

# Dependencies
We have uploaded different model files and configuration files. They need to be run based on MMDetection. 
- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1
  
For more installation details, please refer to  [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/release)

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

# Useage
If you want to train the model, run the following command in the terminal:
```shell
python tools/train.py configs/xxx.py  # xxx.py is the configuration file you want to use
```
If you want to validate the model and generate images, run the following command in the terminal:
```shell
python tools/test.py configs/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```

# Acknowledgement
This project is developed based on the [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/release) and [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main). And here, we would like to thank Chen for his support and the developers of the MMDetection project.

# License
This project is licensed under the [Apache 2.0 license](LICENSE).
