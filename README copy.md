# Lifelong Event Detection via Optimal Transport
Source code for the ACL Rolling Review submission LEDOT.


## Data & Model Preparation

We preprocess the data similar to [Lifelong Event Detection with Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/) (Yu et al., EMNLP 2021), run the following commands to prepare data:
```bash
python prepare_inputs.py
python prepare_stream_instances.py
```

## Training and Testing

To start training on MAVEN, run:
```bash
bash sh/maven.sh
```

## Requirements:
- transformers == 4.23.1
- torch == 1.9.1
- torchmeta == 1.8.0
- numpy == 1.21.6
- tqdm == 4.64.1
- scikit-learn
- cvxpy

## Install steps
Install env python=3.8, torch 1.9.1 first then other libraries 
rm -rf data/
pip install gdown
gdown https://drive.google.com/drive/folders/10eQsBwqXSGkuh9pZ_X_6fQsKG_UDIaPH -O data/ --folder
mkdir log

bash sh/...


gdown https://drive.google.com/drive/folders/1Jd6dHFgvE8xkV6w14spsU6D-Ypq6ZJNM -O data_llm2vec/ --folder
move data to `data_llm2vec`
mkdir -p data_llm2vec/features
