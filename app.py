import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype="float32"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False
)

def answer_question(prompt):
    output = pipe(prompt)[0]["generated_text"]
    return output

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Answer"),
    title="PDF Insight LLM",
    description="Custom LLM backend for your PDF Insight AI app.",
)

# ⭐ THIS IS THE IMPORTANT LINE ⭐
demo.launch(api_name="ask")
