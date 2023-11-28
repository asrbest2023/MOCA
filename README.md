# Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This is the code repository for Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition.
This includes the implementation of MOCA (**M**ulti-hyp**O**theses-based **C**urriculum learning for semi-supervised **A**SR),
our novel approach for semi-supervised automatic speech recognition (ASR).

## Abstract
How can we accurately transcribe speech signals into texts when only a portion of them are annotated?
ASR systems are extensively utilized in many real-world applications including automatic translation systems and transcription services.
Due to the exponential growth of available speech data without annotations and the significant costs of manual labeling, semi-supervised automatic speech recognition (ASR) approaches have garnered attention. 
Such scenarios include transcribing videos in streaming platforms, where a vast amount of content is uploaded daily but only a fraction of them are transcribed manually.
Previous approaches for semi-supervised ASR use a pseudo-labeling scheme to incorporate unlabeled examples during training.
% However, their performance is limited since they fail to consider the uncertainty associated with the pseudo labels when employing them as labels for unlabeled instances.
Nevertheless, their effectiveness is restricted as they do not take into account the uncertainty linked to the pseudo labels when using them as labels for unlabeled cases.
In this paper, we propose MOCA (**M**ulti-hyp**O**theses-based **C**urriculum learning for semi-supervised **A**SR), an accurate framework for semi-supervised ASR.
MOCA generates multiple hypotheses for each speech instance to consider the uncertainty of the pseudo label.
Furthermore, MOCA considers the various degree of uncertainty in pseudo labels across speech instances, enabling a robust training on the uncertain dataset.
Extensive experiments on real-world speech datasets show that MOCA successfully improves the transcription performance of previous ASR models.

## Requirements

We recommend using the following versions of packages:
- `PyYAML>=6.0`
- `tqdm`
- `pandas`
- `numpy`
- `opencv-python`
- `Pillow>=9.4.0`
- `onnxruntime>=1.15.0`

## Data Overview
The datasets are available at (https://drive.google.com/drive/folders/1RSyw6aExar_5Li_j2Jy59q_IfErLkNH1?usp=share_link).

|        **Dataset**        |                  **Path**                   | 
|:-------------------------:|:-------------------------------------------:| 
|       **LJSpeech**        |           `Datasets/LJSpeech-1.1`           | 

## How to Run
You can run the demo script in the directory by the following code.
```
python finetune.py
```
The demo script finetunes a seed model incorporating the unlabeled examples with multiple pseudo labels.
You can reproduce the results in the paper by running the demo script while changing the configuration file (`./configs.py`).

## References
The codes are written based on the `mltu` package (https://github.com/pythonlessons/mltu).