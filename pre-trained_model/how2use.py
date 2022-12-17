"""

"""

import eval_utils

palmtree = eval_utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")

# Tokens have to be seperated by spaces.
text = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "call memcpy", 
        "mov [ rcx + rbx ] 0x0", 
        "mov rcx rax", 
        "mov [ rax ] 0x2e"]

# It is better to make batches as large as possible.
embeddings = palmtree.encode(text)
print(f"{embeddings.shape=}")
print(f"{embeddings=}")
