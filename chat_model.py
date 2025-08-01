from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

model_id = "/home/mony/ai_projects/ai-agent/mistralaiMistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass

conversation = [{"role": "user", "content": "What's the weather like in Paris?"}]
tools = [get_current_weather]


# # format and tokenize the tool use prompt 
# inputs = tokenizer.apply_chat_template(
#             conversation,
#             tools=tools,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
# )
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)



# inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
