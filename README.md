# GPT-2 Implementation

Yugdeep Singh Bhatia
yugdeep.singhbhatia.che20@itbhu.ac.in
Roll No. 20045151
Indian Institute of Technology, Varanasi (BHU)


The file `train__task3.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train_task3.py` is a ~300-line boilerplate training loop and `model_task1.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI.


## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start


```
$ python data/shakespeare_char/prepare.py
```

Time taken: 7 minutes
Best Validation Loss: 2.66

The model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```
$ python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples
