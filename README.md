# SimCAS
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/xjw-nlp/SimCAS/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/xjw-nlp/SimCAS/blob/main/DATA_LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SafeAILab/RAIN/issues) 
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat)](https://github.com/SafeAILab/RAIN/pulls)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

The repository contains the source code, data, and models for the paper [Chunk, Align, Select: A Simple Long-sequence Processing Method for Transformers](https://arxiv.org/abs/2308.13191#), ACL 2024.
## Quick Links
- [Recap](#recap)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
## Recap
In this paper, we propose a simple three-stage framework to process long-sequence input for transformers.
![pipeline](./model.png)
## Installation
- `conda create --name env --file spec-file.txt`
- `pip install -r requirements.txt`
- Using `compare_mt` -> https://github.com/neulab/compare-mt
  ```console
  git clone https://github.com/neulab/compare-mt.git
  cd ./compare-mt
  pip install -r requirements.txt
  python setup.py install
  ```
- For the ROUGE calculation with the standard Perl package from [here](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5).
  ```console
  # make sure perl and cpan is installed
  perl --version
  cpan --version

  # install XML::DOM
  # may need sudo
  sudo cpan XML::DOM
  
  # download ROUGE-1.5.5
  git clone https://github.com/summanlp/evaluation
  
  # ROUGE 1.5.5 can be found in evaluation/ROUGE-RELEASE-1.5.5
  export ROUGE=/absolute/path/to/ROUGE-RELEASE-1.5.5
  
  # Optional: setting environment variable
  echo "export ROUGE=\"${ROUGE}\"" >> ~/.bashrc
  source ~/.bashrc
  
  # modify the db file
  cd ${ROUGE}/data/WordNet-2.0-Exceptions/
  mv WordNet-2.0.exc.db WordNet-2.0.exc.db.bak
  ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
  
  cd $ROUGE
  ./runROUGE-test.pl
  # if there is no error message, then you have successfully installed ROUGE
  ```
- For BERTScore, using evaluation tool from [here](https://github.com/Tiiiger/bert_score)

## Preprocessing
We use the following datasets for our experiments. 
- arXiv -> [https://github.com/armancohan/long-summarization](https://github.com/armancohan/long-summarization)
- PubMed -> [https://github.com/armancohan/long-summarization](https://github.com/armancohan/long-summarization)
- GovReport -> [https://github.com/luyang-huang96/LongDocSum](https://github.com/luyang-huang96/LongDocSum)
- SummScreen -> [https://github.com/mingdachen/SummScreen](https://github.com/mingdachen/SummScreen)
- Multi-News -> [https://github.com/Alex-Fabbri/Multi-News](https://github.com/Alex-Fabbri/Multi-News)
- WCEP -> [https://github.com/allenai/PRIMER](https://github.com/allenai/PRIMER)
- NarrativeQA -> [https://github.com/google-deepmind/narrativeqa](https://github.com/google-deepmind/narrativeqa)

We can also download the preprocessed datasets: [arXiv](https://huggingface.co/datasets/ccdv/arxiv-summarization), [PubMed](https://huggingface.co/datasets/ccdv/pubmed-summarization), [GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization), [SummScreen](), [Multi-News](https://drive.google.com/file/d/15HI5DFiVbSvHnXPOlzOhPAnbPUDdRQxZ/view?usp=sharing), [WCEP](), [NarrativeQA]().
  
## Training
```console
python main.py --cuda --gpuid [list of gpuid] --config [name of config] -l -p [number of port]
```
## Evaluation
### Example on SummScreen
```console
python main.py --cuda --gpuid 0 --config summscreen -e --model_pt summscreen/model_generation.bin

export CLASSPATH=/nas/xiejiawen/stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar
cat ./result/summscreen/test.out | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ./result/summscreen/test.out.tokenized
cat ./result/summscreen/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ./result/summscreen/test.target.tokenized
python cal_rouge.py --ref ./result/summscreen/test.target.tokenized --hyp ./result/summscreen/test.out.tokenized --type summscreen -l

python cal_rouge.py --ref ./result/summscreen/test.target.tokenized --hyp ./result/summscreen/test.out.tokenized --type summscreen -l -p
```
## Citation
```console
@misc{xie2023chunk,
      title={Chunk, Align, Select: A Simple Long-sequence Processing Method for Transformers}, 
      author={Jiawen Xie and Pengyu Cheng and Xiao Liang and Yong Dai and Nan Du},
      year={2023},
      eprint={2308.13191},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
