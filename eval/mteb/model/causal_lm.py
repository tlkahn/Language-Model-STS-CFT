import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

torch.manual_seed(0)
logger = logging.getLogger(__name__)


class CausalLMEncoder:
    def __init__(self, model_path='sarvamai/sarvam-1', adapter_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_eos_token = True
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map='cuda',
                                                          trust_remote_code=True)
        if adapter_path is not None:
            self.model.load_adapter(adapter_path)

    CACHE_CLEAR_INTERVAL = 100

    def _encode_single(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True).hidden_states[-1][0, -1, :]
        emb = out.squeeze().float().cpu().numpy()
        del inputs, out
        return emb

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """
        Returns embeddings for the given sentences.

        Args:
            sentences: List of sentences to encode

        Returns:
            numpy array of shape (len(sentences), hidden_dim)
        """

        out = []
        n = len(sentences)
        logger.info(f"Encoding {n} sentences...")
        start = time.time()

        for i, s in enumerate(tqdm(sentences, desc="Encoding", unit="sent")):
            out.append(self._encode_single(s))
            if (i + 1) % self.CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()

        elapsed = time.time() - start
        rate = n / elapsed if elapsed > 0 else 0
        logger.info(f"Encoded {n} sentences in {elapsed:.1f}s ({rate:.1f} sent/s)")

        return np.array(out)
