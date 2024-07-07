import gradio as gr
import random
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import os


# Set environment variables for OpenMP
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["MKL_NUM_THREADS"] = "40"

# Set the number of threads for PyTorch
torch.set_num_threads(40)
torch.set_num_interop_threads(40)

# Verify the number of threads
print(f"Number of threads: {torch.get_num_threads()}")
print(f"Number of interop threads: {torch.get_num_interop_threads()}")
device1 = 'cpu'
device2 = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained("/data/xiachunwei/Datasets/Models/Meta-Llama-3-8B-Instruct")
model1 = AutoModelForCausalLM.from_pretrained("/data/xiachunwei/Datasets/Models/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).to(device1)
model2 = AutoModelForCausalLM.from_pretrained("/data/xiachunwei/Datasets/Models/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).to(device2)
model1.eval()
model2.eval()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

with gr.Blocks() as demo:
    with gr.Row():
        chatbot1 = gr.Chatbot(label="CPU")
        chatbot2 = gr.Chatbot(label="GPU")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, ""]]

    def bot1(history):
        stop = StopOnTokens()
        messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history])
        print("messages: ", messages)
        model_inputs = tokenizer([messages], return_tensors="pt").to(device1)
        streamer = TextIteratorStreamer(tokenizer, timeout=100., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=1.0,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
            )
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            t = Thread(target=model1.generate, kwargs=generate_kwargs)
            t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                yield [[partial_message, ""]]

    
    def bot2(history):
        stop = StopOnTokens()
        messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history])
        print("messages: ", messages)
        model_inputs = tokenizer([messages], return_tensors="pt").to(device2)
        streamer = TextIteratorStreamer(tokenizer, timeout=100., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=1.0,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
            )
        t = Thread(target=model2.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                yield [[partial_message, ""]]

    msg.submit(user, [msg, chatbot1], [msg, chatbot1], queue=False).then(
        bot1, chatbot1, chatbot1
    )
    clear.click(lambda: None, None, chatbot1, queue=False)
    
    msg.submit(user, [msg, chatbot2], [msg, chatbot2], queue=False).then(
        bot2, chatbot2, chatbot2
    )
    clear.click(lambda: None, None, chatbot2, queue=False)
demo.queue()
demo.launch(share=True)
