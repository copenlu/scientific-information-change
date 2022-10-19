from sentence_transformers import SentenceTransformer, util
from typing import Optional, AnyStr, List
import numpy as np
import torch
import torch.nn.functional as F


class SimilarityEstimator(object):
    """
    Estimator of information matching score (IMS) between two scientific sentences
    """
    def __init__(
            self,
            model_name_or_path: Optional[AnyStr] = 'copenlu/spiced',
            device: Optional[AnyStr] = None,
            use_auth_token: Optional[bool] = False,
            cache_folder: Optional[AnyStr] = None
    ):
        """

        :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name. Defaults to the best model from Wright et al. 2022.
        :param device: Device (like ‘cuda’ / ‘cpu’) that should be used for computation. If None, checks if a GPU can be used.
        :param use_auth_token: HuggingFace authentication token to download private models.
        :param cache_folder: Path to store models
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.use_auth_token = use_auth_token
        self.cache_folder = cache_folder

        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device,
            use_auth_token=use_auth_token,
            cache_folder=cache_folder
        )

    def estimate_ims(
            self,
            a: List[AnyStr],
            b: List[AnyStr]
    ) -> np.ndarray:
        """
        Estimate the information matching score between all sentences in 'a' and all sentences in 'b'. Score will a scalar between 1 and 5, where 1 means no information similarity and 5 means the information is exactly the same between the two sentences.
        :param a: A list of sentences
        :param b: Second list of sentences
        :return: A matrix S of size $N$x$M$ where $N$ is the length of list $a$, $M$ is the length of list $b$, and entry $S_{ij}$ is the information matching score between sentence $a_{i}$ and $b_{j}$
        """
        sentence1_embedding = self.model.encode(a)
        sentence2_embedding = self.model.encode(b)
        S = (util.cos_sim(sentence1_embedding, sentence2_embedding).clip(min=0, max=1) * 4) + 1

        return S.detach().numpy()


    def estimate_ims_array(
            self,
            a: List[AnyStr],
            b: List[AnyStr]
    ) -> List:
        """
        Estimate the information matching score between each sentence in $a$ and its corresponding $b$ (i.e. $a_{i}$ and $b_{i}$). Score will a scalar between 1 and 5, where 1 means no information similarity and 5 means the information is exactly the same between the two sentences.
        :param a: A list of sentences
        :param b: Second list of sentences of the same size as $a$
        :return: A list $s$ of size $N$ where $N$ is the length of both list $a$ and list $b$ and entry $s_{i}$ is the information matching score between $a_{i}$ and $b_{i}$
        """
        assert len(a) == len(b), f"len(a) != len(b), lists of sentences must be equal length. len(a) == {len(a)}, len(b) == {len(b)}"
        sentence1_embedding = self.model.encode(a)
        sentence2_embedding = self.model.encode(b)
        scores = F.cosine_similarity(torch.Tensor(sentence1_embedding), torch.Tensor(sentence2_embedding), dim=1).clip(
            min=0).squeeze().cpu().numpy()
        # Convert to range [1,5], assume anything below 0 is 0
        s = (scores * 4) + 1

        return s.tolist()

