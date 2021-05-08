# CS482/682 Final Project Group 27
## Locally Connected Recurrent Neural Networks for Natural Language Processing

This code was forked from the [GitHub for the following paper:](https://github.com/BeckyMarvin/LM_syneval)

R. Marvin and T. Linzen. 2018. Targeted Syntactic Evaluation of Language Models. Proceedings of EMNLP.

### Locally Connected Layers

Our implementation of Locally Connected Layers lives in `LCRNN/word-language-model/`. For documentation of the locally connected layers, see [LCRNN/locallyconnectedlayer.ipynb](https://github.com/adityayedetore/LCRNN/tree/master/word-language-model). 

### Detailed Syntactic Evaluation 

We provide the iPython Notebook [ModelEval.ipynb](https://github.com/adityayedetore/LCRNN/blob/master/ModelEval.ipynb) for ease of evaluation. 

## HOW TO USE THIS CODE

The following is from the original README for this repo, and ise useful for getting this code up and running.

### Language model training data

We used the same training data as Gulordava et al. (2018). Each corpus consists of around 100M tokens from English Wikipedia. We used training (80M) and validation (10M) subsets in our experiments. All corpora were shuffled at sentence level. Links to download the data are below:

[train](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/train.txt) / [valid](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/valid.txt) / [test](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/test.txt) / [vocab](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt)


### Training the language models

We provide the code used to train both LSTMS.

To train a basic LSTM language model:
```
python main.py --lm_data data/lm_data --save models/lstm_lm.pt --save_lm_data models/lstm_lm.bin
```

To train a multitask LSTM model (jointly trained to do language modeling and CCG supertagging):
```
python main.py --lm_data data/lm_data --ccg_data data/ccg_data --save models/lstm_multi.pt --save_lm_data models/lstm_multi.bin
```

Alternatively, you can train a language model or multitask model by using the `train_lm.sh` or `train_multitask.sh` scripts (found in `example_scripts`). Alternate hyperparameters can be specified in the `hyperparameters.txt` file.

### Testing the language models
PERPLEXITY ONLY: 
Language model:
```
python main.py --test --lm_data data/lm_data --save models/$model_pt --save_lm_data models/$model_bin
```

Multitask model:
```
python main.py --test --lm_data data/lm_data
```


