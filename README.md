# DWT-Mixer: An all-MLP Model via Token Shift
## DWT-Mixer
![Figure 1. The overall architecture of the proposed DWT-Mixer](https://github.com/wyl-privacy-project/DWT-Mixer/blob/main/Figure/DWT-Mixer%20overall.jpg)
## DWT-Mixer token mixing layer
![Figure 2. DWT-Mixer token mixing layer](https://github.com/wyl-privacy-project/DWT-Mixer/blob/main/Figure/DWT.jpg)
## Usage
### Install
- Clone this repo:
```bash
git clone https://github.com/wyl-privacy-project/DWT-Mixer
cd DWT-Mixer
```
- Create a conda virtual environment and activate it:
```bash
conda create -n DWTMixer python=3.8 -y
conda activate DWTMixer
```
## Caching Vocab Hashes(like pNLP-Mixer)

```bash
python projection.py -v=wordpiece/vocab.txt -c=cfg/Config_Path -o=OutPut_File
```
- Config_Path: path to the configurations file
- OutPut_File: path where the resulting file will be saved,default='/vocab.npy')
## Train/Test

```bash
python run.py -c=Config_Path -n=MODEL_NAME -m=MODE -p=CKPT_PATH
```
- Config_Path: path to the configurations file
- MODEL_NAME: model name to be used for pytorch lightning logging
- MODE: train or test
- CKPT_PATH: checkpoint path to resume training from or to use for testing

## Experimental Results
The checkpoints used for evaluation are available [here](https://drive.google.com/drive/folders/18E_o8_Q5EberyKdM8-2aUiz0Ll8Kd0et?usp=drive_link).
### Topic Classification 
|Model|AG News(%)|DBpedia(%)|Params(M)|
|:--:|:--:|:--:|:--:|
| DWT-Mixer-S | 91.63 | 98.30 | 0.157 |
| DWT-Mixer-L | 92.07 | 98.50 | 0.594 |

### Sentiment Analysis

| Model | IMDB(%) | Yelp-2(%) | Amazon-2(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| DWT-Mixer-S | 88.11	| 95.67 |	93.33	| 0.157 |
| DWT-Mixer-L | 88.70	| 96.02	| 93.93 |	0.594 |

###  Natural Language Inference

| Model | SST-2(%) |	CoLA(%) |	QQP(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| DWT-Mixer-S | 83.32 |	71.55	| 81.75	| 0.157 |
| DWT-Mixer-L | 86.26	| 70.38	| 83.01 |	0.594 |
