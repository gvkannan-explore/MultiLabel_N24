import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from dotenv import load_dotenv
from numpy.typing import NDArray
from rich import print
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import RobertaModel, RobertaTokenizer

@dataclass
class ProcessedText:
    """
    Container for processed text data
    """

    cleaned_text: str
    sentences: List[str]
    num_sentences: int

class RoBERTaPreprocessor:
    """
    Housing all the relevant preprocessing steps for RoBERTa within this class - namely:
    1. Cleaning all text while preserving important punctuation and structure.
    2. Splitting text into stenctence while taking care of abbreviations.
    3. Tokenizing and encoding the text.
    """

    def __init__(self, max_length: int = 512) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_length = max_length
        self.abbreviations = {"mr.", "mrs.", "dr.", "st.", "ave.", "prof."}

    def clean_text(self, text: str) -> str:
        """
        Cleaning all text while preserving important punctuation and structure.
        """

        ## Replace multiple newlines/spaces with single space
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        ## Remove URLs and emails:
        text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
        text = re.sub(r"\b[\w-]+@[\w-]+[.][\w-]+", "", text)  # Remove emails

        ## Repalce multiple white-spaces with a single space
        text = " ".join(text.split())

        ## Normalize dashes to hyphen
        text = text.replace("—", "-").replace("–", "-")

        ## Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

        ## Additional cleaning:
        text = re.sub(r"[\u0080-\uFFFF]", "", text)  # Remove non-ASCII characters
        text = re.sub(r"\d+", "NUM", text)  # Replace numbers with "NUM"

        # Remove leading/trailing whitespace
        return text.strip()

    def text2sentences(self, text: str) -> List[str]:
        """
        Split the text into sentences while handling common abbreviations
        """
        sentences = []
        current = []

        words = text.split()  ## Splitting based on whitespaces

        ## Iterate through each word until a stop-character is found.
        for word in words:
            current.append(word)
            if word.lower() in self.abbreviations:
                continue
            if word.endswith((".", "!", "?")):
                sentences.append(" ".join(current))
                current = []

        ## To add the last uncompleted sentence is any:
        if len(current) > 0:
            sentences.append(" ".join(current))

        return sentences

    def process_text(self, text: str) -> ProcessedText:
        """
        Clean & split into sentences
        Returns:
            ProcessedText object with cleaned text and sentences
        """
        cleaned_text = self.clean_text(text)
        sentences = self.text2sentences(cleaned_text)

        return ProcessedText(
            cleaned_text=cleaned_text,
            sentences=sentences,
            num_sentences=len(sentences),
        )

    def encode_for_model(
        self,
        processed_text: ProcessedText,
        add_special_tokens: bool = True,
        truncation: bool = True,    
        padding: str = "max_length",
    ) -> Dict[str, List[int]]:
        """
        Encode processed text for RoBERTa.
        """
        try:
            return self.tokenizer(
                processed_text.cleaned_text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        """
        Process and encode a batch of texts.
        """
        processed_texts = [self.process_text(text).cleaned_text for text in texts]

        return self.tokenizer(
            processed_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            **kwargs,
        )