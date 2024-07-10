import gradio as gr
from transformers import AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from optimum.intel.openvino import OVModelForCausalLM
import torch
from threading import Event, Thread
from uuid import uuid4
from typing import List, Tuple
from langdetect import detect

model_dir = "openvino_model"

def load_model():
    model = OVModelForCausalLM.from_pretrained(model_dir, compile=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

def convert_history_to_token(history: List[Tuple[str, str]], user_input: str):
    text = " ".join([f"User: {item[0]} Assistant: {item[1]}" for item in history])
    text += f" User: {user_input} Assistant:"
    input_token = tokenizer(text, return_tensors="pt").input_ids
    return input_token

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stop_tokens = [tokenizer.eos_token_id]  # Use EOS token as the stopping criteria

def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    user_input = history[-1][0]
    input_ids = convert_history_to_token(history[:-1], user_input)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop_tokens)])

    stream_complete = Event()

    def generate_and_signal_complete():
        model.generate(**generate_kwargs)
        stream_complete.set()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        if len(partial_text) > 20:  # Only detect language if the text length is sufficient
            detected_lang = detect(partial_text)
            if detected_lang == 'en':  # Only consider English responses
                yield history

    stream_complete.wait()

def user(message, history):
    return "", history + [[message, ""]]

def get_uuid():
    """
    Universal unique identifier for the conversation.
    """
    return str(uuid4())

with gr.Blocks() as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown("<h1><center>OpenVINO Chatbot</center></h1>")
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
                container=False,
            )
        with gr.Column():
            submit = gr.Button("Submit")
            stop = gr.Button("Stop")
            clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(
                        label="Temperature",
                        value=0.1,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        interactive=True,
                        info="Higher values produce more diverse outputs",
                    )
                with gr.Column():
                    top_p = gr.Slider(
                        label="Top-p (nucleus sampling)",
                        value=1.0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        interactive=True,
                        info="Sample from the smallest possible set of tokens whose cumulative probability exceeds top_p. Set to 1 to disable and sample from all tokens.",
                    )
                with gr.Column():
                    top_k = gr.Slider(
                        label="Top-k",
                        value=50,
                        minimum=0.0,
                        maximum=200,
                        step=1,
                        interactive=True,
                        info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                    )
                with gr.Column():
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        value=1.1,
                        minimum=1.0,
                        maximum=2.0,
                        step=0.1,
                        interactive=True,
                        info="Penalize repetition — 1.0 to disable.",
                    )
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=2)
demo.launch()