import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory) 

from src.custom_dataset import RawFileDataset
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_file: str = field(default=None, metadata={"help": "train file name"})
    val_file: str = field(default=None, metadata={"help": "val file name"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ### one can customize here, since we set the T for joint loss as 2
        
        batch_input_ids1, batch_input_ids2 = [], []
        batch_attention_mask1, batch_attention_mask2 = [], []
        batch_labels1, batch_labels2 = [], []

        for instance in instances:
            instance1, instance2 = instance["instance_1"], instance["instance_2"]
            batch_input_ids1.append(instance1["input_ids"])
            batch_input_ids2.append(instance2["input_ids"])
            batch_attention_mask1.append(instance1["attention_mask"])
            batch_attention_mask2.append(instance2["attention_mask"])
            batch_labels1.append(instance1["labels"])
            batch_labels2.append(instance2["labels"])
        
        batch_input_ids1 = torch.stack(batch_input_ids1, dim=0)
        batch_input_ids2 = torch.stack(batch_input_ids2, dim=0)
        batch_attention_mask1 = torch.stack(batch_attention_mask1, dim=0)
        batch_attention_mask2 = torch.stack(batch_attention_mask2, dim=0)
        batch_labels1 = torch.stack(batch_labels1, dim=0)
        batch_labels2 = torch.stack(batch_labels2, dim=0)
        
        return {
            "batch_input_ids1": batch_input_ids1,
            "batch_input_ids2": batch_input_ids2,
            "batch_attention_mask1": batch_attention_mask1,
            "batch_attention_mask2": batch_attention_mask2,
            "batch_labels1": batch_labels1,
            "batch_labels2": batch_labels2,
        }
        

class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids1 = inputs.get("batch_input_ids1")
        input_ids2 = inputs.get("batch_input_ids2")
        batch_attention_mask1 = inputs.get("batch_attention_mask1")
        batch_attention_mask2 = inputs.get("batch_attention_mask2")
        batch_labels1 = inputs.get("batch_labels1")
        batch_labels2 = inputs.get("batch_labels2")
        
        outputs1 = model(
            input_ids=input_ids1,
            attention_mask=batch_attention_mask1,
            labels=batch_labels1,
        )
        outputs2 = model(
            input_ids=input_ids2,
            attention_mask=batch_attention_mask2,
            labels=batch_labels2,
        )
        
        outputs = (outputs1, outputs2)
        loss = outputs1.loss + outputs2.loss

        return (loss, outputs) if return_outputs else loss 

              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.pad_token_id = 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    
    train_file = os.path.join(data_args.data_path, data_args.train_file)
    val_file = os.path.join(data_args.data_path, data_args.val_file)
    
    train_dataset = RawFileDataset(training_args, train_file, tokenizer)
    val_dataset = RawFileDataset(training_args, val_file, tokenizer)


    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    model.is_parallelizable = True
    model.model_parallel = True

    trainer = CustomTrainier(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
