import sys
import os
import fire
import torch
import transformers
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from tqdm import *

def main(
    file_path: str = None,
    base_model: str = "/zecheng/model_hub/Llama-2-7b-multinode/checkpoint-1800",
    device: int=None,
    output_dir: str=None,
    max_new_tokens: int=1024,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    
    assert file_path, (
        "Please specify a --file_path, e.g. --file_path='/path/to/json_file'"
    )
    
    assert device is not None, (
        "Please specify a --device, e.g. --device='0'"
    )
    
    assert output_dir, (
        "Please specify a --output_dir, e.g. --output_dir='/path/to/output_dir'"
    )
    
    tokenizer = LlamaTokenizer.from_pretrained("/zecheng/model_hub/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    
    model.half()
    model.eval()
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    def evaluate(
        instruction,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        **kwargs,
    ):
        prompt = instruction
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output
    
    with open(file_path, "r") as f:
        content = [json.loads(line) for line in f]
    
    res = []
    with tqdm(total=len(content)) as pbar:
        for samples in content:
            cond_cate_to_size_pos = samples.get("cond_cate_to_size_pos_seq_modeling")[0]
            cond_cate_size_to_pos = samples.get("cond_cate_size_to_pos_seq_modeling")[0]
            cond_recover_mask_input = samples.get("cond_recover_mask_seq_modeling")[0]
            id_ = samples.get("id_")

          
            cond_cate_to_size_pos = evaluate(cond_cate_to_size_pos, max_new_tokens=max_new_tokens)
            cond_cate_size_to_pos = evaluate(cond_cate_size_to_pos, max_new_tokens=max_new_tokens)
            cond_recover_mask = evaluate(cond_recover_mask_input, max_new_tokens=max_new_tokens)
            
            res.append({
                "cond_cate_to_size_pos": cond_cate_to_size_pos,
                "cond_cate_size_to_pos": cond_cate_size_to_pos,
                "cond_recover_mask": cond_recover_mask,
                "id": id_
            })
            pbar.update(1)
        
    output_file = os.path.join(output_dir, file_path.split("/")[-1])
    
    with open(output_file, "w") as f:
        for line in res:
            f.write(json.dumps(line) + "\n")
            
    print("save done !!")
            
        
if __name__ == "__main__":
    fire.Fire(main)