from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_tokenizer(model_name="answerdotai/ModernBERT-base",
                  bos_token="<BOS>", #beginning of sequence token
                  eos_token="<EOS>", #end of sequence token
                  start_token="<START_ID>", #start of role token for chat template, e.g. <START_ID>user<END_ID> or <START_ID>assistant<END_ID>
                  end_token="<END_ID>", #end of role token for chat template    
                  eot_token="<EOT_ID>"): #end of turn token for chat template
    """
    Load tokenizer, add special tokens, and set up a chat template.
    """

    """
    <BOS><START_ID>user<END_ID>
    What is the capital of France?<EOT_ID><START_ID>assistant<END_ID>
    Paris<EOS>
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Define our special tokens
    special_tokens = {
        "bos_token": bos_token, 
        "eos_token": eos_token,
        "additional_special_tokens": [
            start_token,
            end_token,
            eot_token,
        ]
    }

    # Add them to tokenizer
    tokenizer.add_special_tokens(special_tokens)

    # Set EOS token as PAD token and CLS to BOS Token because we don't want to use the default ones
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token
    
    ### Template Processing for Tokenizing Pretraining Data ### 
    ## This will add BOS and EOS tokens to the beginning and end of the sequence respectively during tokenization
    # <bos> Hello World <eos> -> [bos_token_id, token_id_hello, token_id_world, eos_token_id]
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ],
    )

    ### Chat Template for SFT ###
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token if loop.first else '' }}"
        f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
        "{{ message['content'] }}"
        f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ '{start_token}' + 'assistant' + '{end_token}' }}}}"
        "{% endif %}"
    )

    return tokenizer


def test_tokenizer():
    
    tokenizer = get_tokenizer()
    text = "Hello World"
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    decoded = tokenizer.decode(ids, skip_special_tokens = False)

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"}]
    
    encoded = tokenizer.apply_chat_template(messages, tokenize = True, add_special_tokens = True)
    decoded = tokenizer.decode(encoded, skip_special_tokens = False)

    messages = [
        {"role": "user", "content": "who wrote Hamlet?"},
    ]

    # Add assistant generation prompt
    encoded = tokenizer.apply_chat_template(messages, 
                                            tokenize = True, 
                                            add_special_tokens = True, 
                                            add_generation_prompt = True # This will add <START_ID>assistant<END_ID> 
                                                                         # at the end of the sequence to prompt the model 
                                                                         # to generate a response
                                            )
    

if __name__ == "__main__":
    test_tokenizer()