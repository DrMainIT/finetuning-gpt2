from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, pipeline

tokenizer = GPT2Tokenizer.from_pretrained("erwanf/gpt2-mini")
model = GPT2ForSequenceClassification.from_pretrained("./models/finetuned_gpt2")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

results = classifier(["I love you", "I hate you", "I don't know how I feel"])

print(results)