# Semantic Segmentation and Object Detection on Pytorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

This project aims at providing an easy-to-use, compact implementation framework for training semantic segmentation and detection in PyTorch.

## Usage
### Train
-----------------
- semantic model training
```
#config your model param file in ./config/configs/xxx.py
cd general_frame
python3 main.py train
```
- detection model training
```
#config your model param file in ./config/configs/xxx.py
cd general_frame
python3 main.py train
```

### Evaluation
- semantic model testing
```
#config your model param file in ./config/configs/xxx.py
cd general_frame
python3 main.py test or val
```
- detection model testing
```
#config your model param file in ./config/configs/xxx.py
cd general_frame
python3 main.py test or val
```

```
.{ROOT}
├── config
│   ├── configs
│   │   ├── centernet.py
│   │   ├── erfnet.py
│   │   └── unetpp.py
│   ├── __init__.py
│   └── opts.py
├── datasets
│   ├── augment
│   │   ├── image.py
│   │   ├── __init__.py
│   │   └── transforms.py
│   ├── __init__.py
│   ├── specific_data
│   │   ├── bdd.py
│   │   ├── coco.py
│   │   ├── __init__.py
│   │   ├── lane.py
│   │   └── TT100k.py
│   ├── task
│   │   ├── base.py
│   │   ├── centernet_task.py
│   │   ├── __init__.py
│   │   └── seg_task.py
│   └── test_dataset.py
├── detectors
│   ├── base_detector.py
│   ├── centernetdet.py
│   ├── __init__.py
│   └── unetppdet.py
├── main.py
├── models
│   ├── __init__.py
│   ├── networks
│   │   ├── base_models
│   │   │   └── resnet50.py
│   │   ├── dla.py
│   │   ├── erfnet.py
│   │   └── unetpp.py
│   ├── test_model.py
│   └── utils.py
├── README.md
├── trainers
│   ├── base_trainer.py
│   ├── centernet_trainer.py
│   ├── erfnet_trainer.py
│   ├── __init__.py
│   └── unetpp_trainer.py
└── utils
    ├── data_parallel.py
    ├── __init__.py
    ├── losses.py
    ├── utils.py
    ├── visualizer.py
    └── vis_visualizer.py

```

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
