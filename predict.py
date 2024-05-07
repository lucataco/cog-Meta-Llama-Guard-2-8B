# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
import os
import time
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/hf/meta-llama/meta-llama-guard-2-8b/bb78080332eda00343dc37b0465b43bbf22c0251/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        dtype = torch.bfloat16
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE,
            cache_dir=MODEL_CACHE
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype=dtype,
            device_map="cuda",
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        prompt: str = Input(description="User input", default="I forgot how to kill a process in Linux, can you help?"),
        assistant: str = Input(description="Assistant response to user prompt", default=None),
    ) -> str:
        """Run a single prediction on the model"""
        if isinstance(assistant, str):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ]
            print("Assistant response detected, checking ASSISTANT response")
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            print("No Assistant response detected, checking USER prompt")
        
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return result
