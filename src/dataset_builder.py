import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CUADDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

def prepare_train_features(df, tokenizer_name="roberta-base", max_length=512, doc_stride=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Handle NaN values explicitly from CSV reading
    df["answer_text"] = df["answer_text"].fillna("")
    positive_df = df[df["answer_text"] != ""].reset_index(drop=True)
    
    print(f"Tokenizing {len(positive_df)} positive QA examples with sliding window...")
    
    questions = positive_df["question"].tolist()
    contexts = positive_df["context"].tolist()
    
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        
        sample_index = sample_mapping[i]
        answers = positive_df.iloc[sample_index]
        
        start_char = answers["answer_start"]
        end_char = start_char + len(answers["answer_text"])
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Locate context tokens in window
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # Check if answer span fits within current window
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
            
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    
    print(f"Tokenization complete! Total windowed chunks: {len(tokenized_examples['input_ids'])}")
    
    # Save processed tensors
    dataset = CUADDataset(tokenized_examples)
    os.makedirs("data", exist_ok=True)
    torch.save(dataset, os.path.join("data", "train_dataset.pt"))
    print("Saved processed PyTorch dataset to data/train_dataset.pt")
    
    return dataset

if __name__ == "__main__":
    df = pd.read_csv("data/cuad_flattened.csv")
    prepare_train_features(df)              
      
       
        
         
          
           
            
             

              