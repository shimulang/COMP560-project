import torch, transformers

model_id = "/root/vln/LLaMA-Factory/saves/llama3-70b/lora/sft-megred-model/"

# Free up unused memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16}, #  bfloat16
    device_map="auto"
)

step_descriptions = "\"Moving forward, with horizontal is a room with a couch and a brick wall.\", " \
                    "\"To your 150 degrees right, with horizontal is a living room with a couch and a ceiling fan.\"," \
                    " \"Moving forward, with horizontal is a person sitting on a bench in front of a building.\", " \
                    "\"Moving forward, with horizontal is a brick building with a lot of windows.\", " \
                    "\"Moving forward, with horizontal is a brick building with a glass door on the outside of it.\"," \
                    " \"To your 30 degrees right, with horizontal is a view of a pool from inside a building.\"," \
                    " \"To your 30 degrees right, with horizontal is a view of a patio with benches and a view of a body of water.\""
PLANNER_PROMPT = (
    f"""
                               Instructions::
                                   1) Clarity and Self-containment: Each instruction should be self-contained, providing clear directions from one point to the next without using specific angular measurements.
                                   2) Quotation Marks: Present each instruction within quotation marks to clearly indicate its beginning and end.
                                   3) Spatial Logic: Follow the spatial logic outlined in the descriptions, aiming to mimic the style and detail orientation of example ground truth instructions provided.
                                   4) Observational Basis: Generate instructions based on the observational descriptions provided, which detail objects and spatial relationships.
                                   5) Instructions Length and Format:  Each Instructions must be a self-contained trajectory containing 3 to 5 sentences, and the total word count of all Instructions must be less than 50 words.
                                   don’t repeat environment that has been described in previous sentence.


                               Please generate new instructions based on the  descriptions provided。

                               Begin!
                               Generating New Instructions:
                               Input:{step_descriptions}
                               Output：
                              """
)


messages = [
    {"role": "user", "content": PLANNER_PROMPT},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# for i in range(1000):
outputs = pipeline(
    prompt,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])