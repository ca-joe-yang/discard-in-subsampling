# Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time

### [Project Page]() | [Paper]()

This is the official implementation of our paper: "Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time" accepted in ECCV 2024.

## Abstract
Subsampling layers play a crucial role in deep nets by discarding a portion of an activation map to reduce its spatial dimensions. This encourages the deep net to learn higher-level representations. Contrary to this motivation, we hypothesize that the discarded activations are useful and can be incorporated on the fly to improve models' prediction. To validate our hypothesis, we propose a search and aggregate method to find useful activation maps to be used at test-time. We applied our approach to the task of image classification and semantic segmentation. Extensive experiments over nine different architectures on ImageNet, CityScapes, and ADE20K show that our method consistently improves model test-time performance. Additionally, it complements existing test-time augmentation techniques to provide further performance gains.

## Dependencies
You can set up the environment using the provided script. 
```bash
bash scripts/tools/create_env.sh
```

## Data
Please follow the instructions in [Datasets Preparation](Preparation.md).

## Quick Demo
- For a simple demo of how our modified subsampling layer work, please run the following code:
```bash
python3 test_toy.py
```

## Main results on ImageNet Classification

### Our method compared to TTA methods

```bash
bash scripts/cls/test/without_tta.sh
```

### Our method on its own

```bash
bash scripts/cls/test/with_tta.sh
```

## LICENSE
- This work is licensed under the Apache-2.0 license.
- Our project also involves the following assets from other research or projects.
    1. [ClassTTA, AugTTA](https://github.com/divyashan/test-time-augmentation)
    2. [TIMM](https://github.com/huggingface/pytorch-image-models)
    3. [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

## Citation
```bash
@inproceedings{zheng2024learning,
  title={Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time},
  author={Yang, Chiao-An, Ziwei Liu, and Yeh, Raymond A},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Contact
Please contact Chiao-An Yang [yang2300@purdue.edu] if you have any questions.













