import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_sentence_with_word(initial_text, word):

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)


    prompt = (f"Given the following context: '{initial_text}', generate a coherent and grammatically correct sentence that includes the word '{word}' and maintains the original meaning without introducing new concepts or details.\n")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")


    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)


    first_sentence_with_word = None
    while not first_sentence_with_word:

        output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.85, top_p=0.92, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        generated_text = ' '.join(generated_text.split())
        sentences = generated_text.split('.')
        word_lower = word.lower()

        first_sentence_with_word = next((sentence for sentence in sentences if word_lower in sentence.lower()), None)

        if first_sentence_with_word:
          return first_sentence_with_word.strip() + '.'

# Example usage
initial_text = "Today is a wonderful day to learn python."
word = "python"
print(generate_sentence_with_word(initial_text, word))