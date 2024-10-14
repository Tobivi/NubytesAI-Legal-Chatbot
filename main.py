from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr, validator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Dict
from dotenv import load_dotenv
import os
import asyncio
from huggingface_hub import login
import uvicorn
import re
from fastapi.responses import StreamingResponse
from threading import Thread
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("Nubytes05/NubytesAI", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "Nubytes05/NubytesAI",
    use_auth_token=hf_token
)

def determine_prompt_type(user_prompt: str) -> str:
    if "court case" in user_prompt.lower() or " v " in user_prompt:
        return "case"
    elif "compare" in user_prompt.lower():
        return "multi_jurisdiction"
    else:
        return "subject_matter"

# def contains_general_knowledge(response: str) -> bool:
#     general_knowledge_indicators = [
#         "generally", "it is known", "commonly", "in general", "typically", "often", 
#         "people say", "it is believed", "one might", "one could", "it is possible",
#         "common practice", "well-known fact", "as we know", "for example", "historically"
#     ]
#     return any(phrase in response.lower() for phrase in general_knowledge_indicators)

async def generate_structured_response(user_prompt: str):
    base_instruction = "[INST] You are an AI trained on a specific legal dataset. Only use information from this dataset. If you don't have specific training data on this topic, respond with 'I don't have specific training data on this topic.' Do not use any general knowledge. "
    
    prompt_type = determine_prompt_type(user_prompt)
    try:
        if prompt_type == "subject_matter":
            prompts = {
                "Introduction": f"<s>{base_instruction} Explain the concept of the subject matter in detail within the jurisdiction: {user_prompt} [/INST]",
                "Rule": f"<s>{base_instruction} State the legal rules guiding those principles, mentioning statutes, US codes, cases, and judgments: {user_prompt} [/INST]",
                "Cases": f"<s>{base_instruction} List and provide a detailed analysis of cases dealing with the subject matter in the jurisdiction, including issues, rules and authority, and conclusions: {user_prompt} [/INST]",
                "Summary": f"<s>{base_instruction} Summarize the subject matter based on the discussions above: {user_prompt} [/INST]"
            }
        elif prompt_type == "case":
            prompts = {
                "Issues": f"<s>{base_instruction} Discuss the issue the case deals with: {user_prompt} [/INST]",
                "Case Journey": f"<s>{base_instruction} Discuss the journey of the case, highlighting the courts it had been to and their decisions: {user_prompt} [/INST]",
                "Rules and Authority": f"<s>{base_instruction} State the rules and authority relied upon by the highest court in this case: {user_prompt} [/INST]",
                "Similar Cases": f"<s>{base_instruction} Identify and analyze other cases dealing with similar legal issues or principles as those in the case provided, without reiterating the details of the initial case. Provide detailed analyses including issues, rules, and conclusions: {user_prompt} [/INST]",
                "Conclusion": f"<s>{base_instruction} Conclude by discussing the decision of the case: {user_prompt} [/INST]"
            }
        elif prompt_type == "multi_jurisdiction":
            prompts = {
                "Subject Matter": f"<s>{base_instruction} Discuss the subject matter within each of the selected jurisdictions: {user_prompt} [/INST]",
                "Legal Rules": f"<s>{base_instruction} Discuss the legal rules within each jurisdiction, citing statutes/codes and judgments: {user_prompt} [/INST]",
                "Comparison": f"<s>{base_instruction} Summarize the discussion by comparing the subject matter across the jurisdictions: {user_prompt} [/INST]"
            }
        else:
            raise ValueError("Unknown prompt type")
        
        for key, prompt in prompts.items():
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            thread = Thread(target=model.generate,
                            kwargs={"input_ids": inputs['input_ids'],
                                    "streamer": streamer,
                                    "max_new_tokens": 1024})
            thread.start()
            yield f"{key}:\n"
            for new_text in streamer:
                yield new_text
                await asyncio.sleep(0.1)
            thread.join()
            # if contains_general_knowledge(response):
            #     response = "I don't have specific training data on this topic."
            # yield response + "\n"
    except Exception as e:
        yield f"Error: {str(e)}\n"

class RequestModel(BaseModel):
    user_prompt: constr(min_length=1, max_length=200)
    
    @validator("user_prompt")
    def validate_user_prompt(cls, value):
        sanitized_value = re.sub(r'[^\w\s.,!?]', '', value)
        if not sanitized_value:
            raise ValueError("User prompt is invalid after sanitization")
        return sanitized_value

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Legal Content Generator API. Use the /generate endpoint to get responses based on your prompts."}

@app.post("/generate")
async def generate_response(request_model: RequestModel):
    user_prompt = request_model.user_prompt
    async def response_generator():
        async for response in generate_structured_response(user_prompt):
            yield response
    return StreamingResponse(response_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)