from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token_id = tokenizer.eos_token_id

def text(prompt, max_length=300):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long() 
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.5,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        attention_mask=attention_mask, 
        pad_token_id=tokenizer.eos_token_id  
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = input("Type your sentence here !! ")
    print("Input Prompt:", prompt)
    print("\nGenerated Text:")
    print(text(prompt))
