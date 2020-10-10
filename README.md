# Data-to-text Generation with Style Imitation

*Code to be further cleaned up soon...*

This repo contains the code of the following paper:

[Data-to-text Generation with Style Imitation](https://arxiv.org/abs/1901.09501)

*Shuai Lin, Wentao Wang, Zichao Yang, Xiaodan Liang, Eric P. Xing, Zhiting Hu.*    
*Findings of EMNLP 2020*  
## Requirements

The code has been tested on:
 - `Python 3.6.0 and Python 3.7.6`
 - `tensorflow-gpu==1.14.0`
 - `texar-tf==0.2.1`
 - `cuda 10.0`
 
** NOTE **: 
Due to some compatibility issues, the repo is only compatible by installing `texar==0.2.1` from source, which is as follows:
```bash
wget https://github.com/asyml/texar/archive/v0.2.1.zip
cd texar && pip install .
```
 
Run the following command:

```bash
pip3 install -r requirements.txt
```

### For IE

If you'd like to evaluate the IE for the NBA dataset after training, you need to install Lua Torch, and download the IE models from [here](https://drive.google.com/file/d/1hV8I9tvoL3943OqqPkLFIbTYfFSqsV1e/view?usp=sharing) then unzip the files directly under the directory `data2text/`.


## Data Preparation

The dataset developed in the paper is in the [repo](https://github.com/ZhitingHu/text_content_manipulation). 
Clone the repo and move them into the current directory as:
```
git clone https://github.com/ZhitingHu/text_content_manipulation
cd text_content_manipulation/
mv nba_data/ ../nba_data
mv e2e_data/ ../e2e_data
```

## Training

The following command illustrates how to run an experiment:

```bash
python3 main_ours.py --copy_x --rec_w 0.8 --coverage --exact_cover_w 2.5 --dataset [nba/e2e] --save_path [nba_ours/e2e_ours]
```

Where `[SAVE_PATH]` is the directory you'd like to store all the files related to your experiment, e.g. `my_expr`.

Note that the code will automatically restore from the previously saved latest checkpoint if it exists.

You can start Tensorboard in your working directory and watch the curves and `BLEU(y_gen, y')`.


For the MAST baseline:
```bash
python3 main_baseline.py --bt_w 0.1 --dataset [nba/e2e] --save_path [nba_MAST/e2e_MAST]
```

For the AdvST baseline:
```bash
python3 main_baseline.py --bt_w 1 --adv_w 0.5 --dataset [nba/e2e] --save_path [nba_AdvST/e2e_AdvST]
```

## Inference

### Evaluate IE scores

After trained your model, you may want to evaluate IE (Information Retrieval) scores. The following command illustrates how to do it:

```bash
CUDA_VISIBLE_DEVICES=[GPU-ID] python3 ie.py --gold_file nba_data/gold.[STAGE].txt --ref_file nba_data/nba.sent_ref.[STAGE].txt [SAVE_PATH]/ckpt/hypo*.test.txt
```

which needs about 5 GPUS to run IE models for all `[SAVE_PATH]/ckpt/hypo*.test.txt`. `[STAGE]` can be val or test depending on which stage you want to evaluate. The result will be appended to `[SAVE_PATH]/ckpt/ie_results.[STAGE].txt`, in which the columns represents training steps, `BLEU(y_gen, y')`, IE precision, IE recall, simple precision and simple recall (you don't have to know what simple precision/recall is), respectively.

## Evaluate content scores

After trained your model, you may want to evaluate two content scores via Bert classifier. This simplified model is devised from [the Texar implementation of BERT](https://github.com/asyml/texar/tree/master/examples/bert#use-other-datasetstasks). To evaluate the content fidelity, we simply concatenate each record of `x` or `x'` with `y` and classify whether `y` express the record. In this way, we construct the data in `../bert/E2E` to train the Bert classifier. 

### Prepare data
Run the following cmd to prepare data for evaluation:

```bash
python3 prepare_data.py --save_path [SAVE_PATH] --step [STEP]
[--max_seq_length=128]
[--vocab_file=bert_config/all.vocab.txt]
[--tfrecord_output_dir=bert/E2E] 
```
which processes the previous `[SAVE_PATH]/ckpt/hypos[STEP].valid.txt` into the above mentioned `x | y` fomat in TFRecord data files. Here:

* `max_seq_length`: The maxium length of sequence. This includes BERT special tokens that will be automatically added. Longer sequence will be trimmed.
* `vocab_file`: Path to a vocabary file used for tokenization. 
* `tfrecord_output_dir`: The output path where the resulting TFRecord files will be put in. Be default, it is set to `bert/E2E`.


### Restore and evaluate
We provide a pretrained transformer classifier model [link](https://drive.google.com/drive/folders/1jNaJ_R_f89G8xbAC8iwe49Yx_Z-LXr0i), which achieves 92% accuracy on the test set. Make sure that the pretrained model is put into the `bert/classifier_ckpt/ckpt` directory. Before the evaluation for content, remember to modify the file name of `config_data.py` manually. Then, run the following command to restore and compute the two content scores:

```bash
cd bert/
python3 bert_classifier_main.py  --do_pred --config_data=config_data --checkpoint=classifier_ckpt/ckpt/model.ckpt-13625
[--output_dir=output_dir/]
```
The cmd prints the two scores and the output is by default saved in `output/results_*.tsv`, where each line contains the predicted label for each instance.

# Demo

We provide a `content rewriter` demo based on our NLP toolkit [Forte](https://github.com/asyml/forte/tree/master/examples/content_rewriter), which can
 be visulized with [Stave](https://github.com/asyml/stave/blob/master/src/plugins/dialogue_box/READEME.md) as follows:
 
<p align="center">
<img src="figures/demo.png" width="80%" height="60%" />
</p>