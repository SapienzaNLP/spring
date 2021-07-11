# SPRING

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-spring-to-rule-them-both-symmetric-amr/amr-parsing-on-ldc2017t10)](https://paperswithcode.com/sota/amr-parsing-on-ldc2017t10?p=one-spring-to-rule-them-both-symmetric-amr)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-spring-to-rule-them-both-symmetric-amr/amr-parsing-on-ldc2020t02)](https://paperswithcode.com/sota/amr-parsing-on-ldc2020t02?p=one-spring-to-rule-them-both-symmetric-amr)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-spring-to-rule-them-both-symmetric-amr/amr-to-text-generation-on-ldc2017t10)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2017t10?p=one-spring-to-rule-them-both-symmetric-amr)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-spring-to-rule-them-both-symmetric-amr/amr-to-text-generation-on-ldc2020t02)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2020t02?p=one-spring-to-rule-them-both-symmetric-amr)

This is the repo for [SPRING (*Symmetric ParsIng aNd Generation*)](https://ojs.aaai.org/index.php/AAAI/article/view/17489), a novel approach to semantic parsing and generation, presented at AAAI 2021.

With SPRING you can perform both state-of-the-art Text-to-AMR parsing and AMR-to-Text generation without many cumbersome external components.
If you use the code, please reference this work in your paper:

```
@inproceedings{bevilacqua-etal-2021-one,
    title = {One {SPRING} to Rule Them Both: {S}ymmetric {AMR} Semantic Parsing and Generation without a Complex Pipeline},
    author = {Bevilacqua, Michele and Blloshmi, Rexhina and Navigli, Roberto},
    booktitle = {Proceedings of AAAI},
    year = {2021}
}
```

## Pretrained Checkpoints

Here we release our best SPRING models which are based on the DFS linearization.

### Text-to-AMR Parsing
- Model trained in the AMR 2.0 training set: <a href="http://nlp.uniroma1.it/AMR/AMR2.parsing-1.0.tar.bz2">AMR2.parsing-1.0.tar.bz2</a>

- Model trained in the AMR 3.0 training set: [AMR3.parsing-1.0.tar.bz2](http://nlp.uniroma1.it/AMR/AMR3.parsing-1.0.tar.bz2)

### AMR-to-Text Generation
- Model trained in the AMR 2.0 training set: [AMR2.generation-1.0.tar.bz2](http://nlp.uniroma1.it/AMR/AMR2.generation-1.0.tar.bz2)

- Model trained in the AMR 3.0 training set: [AMR3.generation-1.0.tar.bz2](http://nlp.uniroma1.it/AMR/AMR3.generation-1.0.tar.bz2)


If you need the checkpoints of other experiments in the paper, please send us an email.

## Installation
```shell script
cd spring
pip install -r requirements.txt
pip install -e .
```

The code only works with `transformers` < 3.0 because of a disrupting change in positional embeddings.
The code works fine with `torch` 1.5. We recommend the usage of a new `conda` env.

## Train
Modify `config.yaml` in `configs`. Instructions in comments within the file. Also see the [appendix](docs/appendix.pdf).

### Text-to-AMR
```shell script
python bin/train.py --config configs/config.yaml --direction amr
```
Results in `runs/`

### AMR-to-Text
```shell script
python bin/train.py --config configs/config.yaml --direction text
```
Results in `runs/`

## Evaluate
### Text-to-AMR
```shell script
python bin/predict_amrs.py \
    --datasets <AMR-ROOT>/data/amrs/split/test/*.txt \
    --gold-path data/tmp/amr2.0/gold.amr.txt \
    --pred-path data/tmp/amr2.0/pred.amr.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
```
`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction file (`data/tmp/amr2.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr2.0/pred.amr.txt \
    --out data/tmp/amr2.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models
```
To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation, which provide
results that are around ~0.3 Smatch points lower than those returned by `bin/predict_amrs.py`.

### AMR-to-Text
```shell script
python bin/predict_sentences.py \
    --datasets <AMR-ROOT>/data/amrs/split/test/*.txt \
    --gold-path data/tmp/amr2.0/gold.text.txt \
    --pred-path data/tmp/amr2.0/pred.text.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
```
`gold.text.txt` and `pred.text.txt` will contain, respectively, the concatenated gold and the predictions.
For BLEU, chrF++, and Meteor in order to be comparable you will need to tokenize both gold and predictions using [JAMR tokenizer](https://github.com/redpony/cdec/blob/master/corpus/tokenize-anything.sh).
To compute BLEU and chrF++, please use `bin/eval_bleu.py`. For METEOR, use https://www.cs.cmu.edu/~alavie/METEOR/ .
For BLEURT don't use tokenization and run the eval with `https://github.com/google-research/bleurt`. Also see the [appendix](docs/appendix.pdf).

## Linearizations
The previously shown commands assume the use of the DFS-based linearization. To use BFS or PENMAN decomment the relevant lines in `configs/config.yaml` (for training). As for the evaluation scripts, substitute the `--penman-linearization --use-pointer-tokens` line with `--use-pointer-tokens` for BFS or with `--penman-linearization` for PENMAN.

## License
This project is released under the CC-BY-NC 4.0 license (see `LICENSE`). If you use SPRING, please put a link to this repo.

## Acknowledgements
The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE](http://mousse-project.org) No. 726487 and the [ELEXIS project](https://elex.is/) No. 731015 under the European Union’s Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.
