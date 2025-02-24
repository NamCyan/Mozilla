
# new env requires
# start with converted origin
from typing import Any, List, Tuple, Union, Callable
from transformers import BertTokenizerFast
import json
import os
from tqdm import tqdm
import torch
import numpy as np
import transformers
import random as rd
import h5py

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# export HF_HOME=.cache/
os.environ['HF_HOME'] = '.cache/'
os.environ["HF_TOKEN"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

rd.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class Instance(object):
    '''
    - piece_ids: L
    - label: 1
    - span: 2
    - feature_path: str
    - sentence_id: str
    - mention_id: str
    '''
    def __init__(self, piece_ids:List[int], label:int, span:Tuple[int, int], feature_path:str, sentence_id:str, mention_id:str) -> None:
        self.piece_ids = piece_ids
        self.label = label
        self.span = span
        self.feature_path = feature_path
        self.sentence_id = sentence_id
        self.mention_id = mention_id

    def todict(self,):
        return {
            "piece_ids": self.piece_ids,
            "label": self.label,
            "span": self.span,
            "feature_path": self.feature_path,
            "sentence_id": self.sentence_id,
            "mention_id": self.mention_id
        }


class MAVENPreprocess(object):

    def __init__(self, root, feature_root, tokenizer, l2v, label_start_offset=1, max_length=512, 
                 expand_context=False, split_valid=True, batch_size=32):
        super().__init__()
        self.out = h5py.File(os.path.join(feature_root, "data.h5"), "w")
        self.out.close()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expand_context = expand_context
        self.label_start_offset = label_start_offset
        self.label_ids = {}
        self.collected = set()
        self.model = l2v
        self._sentence_buffer = []
        self.feature_root = feature_root
        self.batch_size = batch_size
        self._file("data_llm2vec/ACE/train.origin", "train")
        self._file("data_llm2vec/ACE/dev.origin", "valid")
        self._file("data_llm2vec/ACE/test.origin", "test")


    def add_sentence(self, sentence_id, piece_ids):
        self.collected.add(sentence_id)
        feature_path = os.path.join(self.feature_root, sentence_id)
        if not os.path.exists(f"{feature_path}.npy"):
            self._sentence_buffer.append((piece_ids, sentence_id))
        if len(self._sentence_buffer) >= self.batch_size:
            self.clear_sentences()

    def clear_sentences(self,):
        if len(self._sentence_buffer) == 0:
          return
        with torch.no_grad():
            self.model.eval()
            sentences = [t[0] for t in self._sentence_buffer]
            sentence_ids = [t[1] for t in self._sentence_buffer]
            length = [len(t) for t in sentences]
            max_l = max(length)
            masks = torch.FloatTensor([[1] * len(t) + [0] *  (max_l - len(t)) for t in sentences])
            sentences = torch.LongTensor([t + [0] * (max_l - len(t)) for t in sentences])
            outputs = self.model(input_ids=sentences.to(torch.device(device)), attention_mask=masks.to(torch.device(device)))
            for i, (s_id, s_l) in enumerate(zip(sentence_ids, length)):
                feature_path = os.path.join(self.feature_root, s_id)
                features = outputs[0][i, :s_l, :]
                # convert 32 bit for features
                features = features.to(dtype=torch.float32)
                features = features.cpu().numpy()
                with h5py.File(os.path.join(self.feature_root, "data.h5"), "a") as out:
                    dataset = out.create_dataset(feature_path, features.shape, dtype="float32")
                    dataset[:] = features
        self._sentence_buffer.clear()


    def _file(self, file_path, split:str):
        with open(file_path, "rt") as fp:
            for instance_line in tqdm(fp):
                instance = json.loads(instance_line)
                if instance["sentence_id"] not in self.collected:
                    self.add_sentence(instance["sentence_id"], instance["piece_ids"])
        self.clear_sentences()


    def _context(self, sentences:List[List[str]]) -> List[Tuple[List[int], int, int]]:
        raise NotImplementedError


    @classmethod
    def _transform_single(cls, token_ids: Union[List[List[str]], List[str], str], spans: Union[List[int], Tuple[int]], tokenizer: BertTokenizerFast, is_tokenized: bool=False) -> Tuple[List[int], List[int]]:
        def _token_span(cls, offsets, s, e):
            ts = []
            i = 0
            while offsets[i][0] <= s:
                i += 1
            ts.append(i - 1)
            i -= 1
            while offsets[i][1] <= e:
                i += 1
            ts.append(i)
            return tuple(ts)
        sent_id = hs = he = ts = te = 0
        _token_ids = _spans = []
        if len(spans) == 4:
            hs, he, ts, te = spans
        else:
            sent_id, hs, he, ts, te = spans
        if isinstance(token_ids, str):
            if is_tokenized:
                raise TypeError("Cannot process single string when 'is_tokenized = True'.")
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                _token_ids = tokens["input_ids"]
                offsets = tokens["offset_mapping"][1:-1]
                h = _token_span(offsets, hs, he)
                t = _token_span(offsets, ts, te)
                _spans = [h[0] + 1, h[1] + 1, t[0] + 1, t[1] + 1]
        elif isinstance(token_ids, List):
            if is_tokenized:
                tokens = tokenizer(token_ids, is_split_into_words=True, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    _token_ids = tokens["input_ids"]
                    offsets = tokens["offset_mapping"]
                    token2piece = []
                    piece_idx = 1
                    for x, y in offsets[1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                else:
                    token2piece = []
                    piece_idx = 1
                    for x, y in tokens["offset_mapping"][sent_id][1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t - 1 + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    offsets = tokens["offset_mapping"][sent_id][1:-1]
                    h = _token_span(offsets, hs, he)
                    t = _token_span(offsets, ts, te)
                    _spans = [h[0], h[1], t[0], t[1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
                else:
                    raise TypeError("Cannot process list of lists of sentences (list of paragraphs).")

        return _token_ids, _spans


def main():


    MAVEN_PATH = "data_llm2vec/ACE" # path for original maven dataset
    feature_root = "data_llm2vec/ace_features"
    os.makedirs(feature_root, exist_ok=True)
    # if os.path.exists('./bert-large-cased'):
    #     bt = BertTokenizerFast.from_pretrained("./bert-large-cased")
    # else:
    #     bt = BertTokenizerFast.from_pretrained("bert-large-cased")
    bt = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Loading MNTP (Masked Next Token Prediction) model.
    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )

    # Wrapper for encoding and pooling operations
    MAVENPreprocess(MAVEN_PATH, feature_root, tokenizer=bt, l2v=model)

if __name__ == "__main__":
    main()