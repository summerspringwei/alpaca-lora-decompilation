import gradio as gr
import random
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

device1 = 'cpu'
device2 = 'cuda:0'


import gradio as gr
import random
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        print("user: user_message: ", user_message)
        print("user: history: ", history)
        return "", history + [[user_message, ""]]

    def bot(history):
        print("bot: history: ", history)
        messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history])
        print("messages: ", messages)
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.5)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch(share=True)


# with gr.Blocks() as demo:
#     with gr.Row():
#         chatbot1 = gr.Chatbot(label="CPU")
#         chatbot2 = gr.Chatbot(label="GPU")
        
#     msg = gr.Textbox()
#     clear = gr.Button("Clear")

#     def user(user_message, history):
#         print("user_message: ", user_message)
#         print("history: ", history)
#         return user_message, history

#     def bot1(message, history):
#         print("message: ", message)
#         print("history: ", history)

#         return message, history
    
#     def bot2(message, history):
#         print("message: ", message)
#         print("history: ", history)

#         return message, history

#     msg.submit(user, [msg, chatbot1], [msg, chatbot1], queue=False).then(
#         bot1, chatbot1, chatbot1
#     )
#     clear.click(lambda: None, None, chatbot1, queue=False)
    
#     msg.submit(user, [msg, chatbot2], [msg, chatbot2], queue=False).then(
#         bot2, chatbot2, chatbot2
#     )
#     clear.click(lambda: None, None, chatbot2, queue=False)
# demo.queue()
# demo.launch(share=True)