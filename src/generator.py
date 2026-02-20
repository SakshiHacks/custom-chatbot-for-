from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # safer for CPU
    low_cpu_mem_usage=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("LLM Loaded Successfully!")

def generate_answer(context, question):
    prompt = f"""
You are a document-based assistant.

Answer only using the given context.
If answer is not found, say:
"Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response