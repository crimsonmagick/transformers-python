import os

import torch

import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer


def main():
    print("is cuda available? cudaAvailable=", torch.cuda.is_available())
    model_path = os.environ.get("LLAMA_MODELS") + "\\llama-2-7b-chat-hf"
    model = LlamaForCausalLM.from_pretrained(model_path)

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline("text-generation",
                                     model=model,
                                     tokenizer=tokenizer,
                                     torch_dtype=torch.float16,
                                     device_map="auto")
    sequences = pipeline(
        'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=400,
    )

    for seq in sequences:
        print(f"{seq['generated_text']}")


if __name__ == "__main__":
    main()
