# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import sys
sys.path.append('/project/sds-rise/weili/Project_2/LLMTemporal2')

# from graphgpt.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

from graphgpt.train.train_graph import train

import logging
logging.basicConfig(filemode='w',filename='logg1.txt', level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s')


if __name__ == "__main__":
    train()
