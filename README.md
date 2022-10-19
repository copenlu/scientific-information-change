# Scientific Information Change
Lightweight package for measuring the information matching score (IMS) between two scientific sentences. This is a measure of the similarity in the information contained in the _findings_ of two different scientific sentences. Useful for research in science communication for matching similar findings as described between scientific papers, news media, and social media at scale and analyzing these findings.

The code and models in this repo come from the following paper:

>Dustin Wright*, Jiaxin Pei*, David Jurgens, and Isabelle Augenstein. 2022. Modeling Information Change in Science Communication with Semantically Matched Paraphrases. In Proceedings of EMNLP 2022. Association for Computational Linguistics.

Please use the following bibtex when referencing this work:

```
@article{modeling-information-change,
      title={{Modeling Information Change in Science Communication with Semantically Matched Paraphrases}},
      author={Wright, Dustin and Jiaxin, Pei and Jurgens, David and Augenstein, Isabelle},
      year={2022},
      booktitle = {Proceedings of EMNLP},
      publisher = {Association for Computational Linguistics},
      year = 2022
}

```

## Installation

Install directly using `pip`:

```
pip install scientific-information-change
```

### Dependencies

```
python>=3.6.0
torch>=1.10.0
sentence-transformers>=2.2.2
numpy
```

If you wish to use CUDA to accelerate inference, install torch with cuda enabled (see https://pytorch.org/)

## Usage

Import the IMS estimator as follows:

```
from scientific_information_change.estimate_similarity import SimilarityEstimator
```

Create the estimator as follows:

```
estimator = SimilarityEstimator()
```

**Note: During your first usage, the package will download a model file automatically, which is about 500MB.**

The similarity estimator takes the following arguments:

```
:param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name. Defaults to the best model from Wright et al. 2022.
:param device: Device (like ‘cuda’ / ‘cpu’) that should be used for computation. If None, checks if a GPU can be used.
:param use_auth_token: HuggingFace authentication token to download private models.
:param cache_folder: Path to store models
```

If you create the estimator with no arguments, it will default to the best trained model from our EMNLP 2022 paper (`copenlu/spiced` in Huggingface). This is an SBERT model pretrained on a large corpus of >1B sentence pairs and further fine-tuned on SPICED. The model will be run on the best available device (GPU if available)

The estimator has two methods for measuring the IMS between sentences. `estimate_ims` takes a list of sentences $a$ of length $N$ and a list of sentences $b$ of length $M$ and returns a numpy array $S$ of size $N \times M$, where $S_{ij}$ is the IMS between $a_{i}$ and $b_{j}$. For example:

```
estimator.estimate_ims(
      a = [
            'Higher-income professionals had less tolerance for smartphone use in business meetings.',
            'Papers with shorter titles get more citations #science #metascience #sciencemetrics'
      ],
      b = [
           'We are intrigued by the result that professionals with higher incomes are less accepting of mobile phone use in meetings.',
           'Allowing users to retract recently posted comments may help minimize regret.',
           'Our analysis suggests that papers with shorter titles do receive greater numbers of citations.'
      ]
   )

>>> returns: array([[4.370547 , 1.042849 , 1.       ],
                    [1.1089203, 1.       , 4.596382 ]], dtype=float32)
```

`estimate_ims_array` takes two lists of sentences of the same length $N$, and returns a list of length $N$ measuring the IMS between corresponding sentences in each list:

```
estimator.estimate_ims_array(
      a = [
            'Higher-income professionals had less tolerance for smartphone use in business meetings.',
            'Papers with shorter titles get more citations #science #metascience #sciencemetrics'
      ],
      b = [
           'We are intrigued by the result that professionals with higher incomes are less accepting of mobile phone use in meetings.',
           'Our analysis suggests that papers with shorter titles do receive greater numbers of citations.'
      ]
   )

>>> returns: [4.370546817779541, 4.596382141113281]
```

