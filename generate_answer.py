
from transformers import pipeline

print("Loading the AI model. Please wait...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

print("AI model loaded successfully!")

context = input(
    "\nPaste relevant text from your heritage documents:\n"
)

question = input(
    "\nEnter your heritage restoration question: "
)

prompt = f"""<|im_start|>system
You are a heritage restoration assistant.
Answer using only the information in the provided context.
If the context does not contain enough information, say so.
<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}
<|im_end|>
<|im_start|>assistant
"""

result = generator(
    prompt,
    max_new_tokens=150,
    do_sample=False,
    return_full_text=False
)

print("\n--- AI Generated Answer ---")
print(result[0]["generated_text"].strip())
