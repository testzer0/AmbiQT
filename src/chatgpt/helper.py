import openai
import time
from chatgpt.credentials import API_KEY

openai.api_key = API_KEY
model = "gpt-3.5-turbo"

def get_reply(directive, prompt, past_interactions=[]):
    messages = [{
        "role": "system",
        "content" : directive
    }]
    for user, cgpt in past_interactions:
        messages.append({
            "role": "user",
            "content": user,
        })
        messages.append({
            "role": "assistant",
            "content": cgpt
        })
    if prompt is not None:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response['choices'][0]['message']['content']

def get_reply_with_retries(directive, prompt, past_interactions=[], \
    backoff_stepsize=10, max_backoff=120, max_tries=30):
    current_backoff = 0
    n = 0
    while True:
        try:
            reply = get_reply(directive, prompt, past_interactions=past_interactions)
            return reply
        except:
            current_backoff = min(current_backoff+backoff_stepsize, max_backoff)
            n += 1
            if n > max_tries:
                print("Tried 30 times, aborting.")
                exit(0)
            print("[Retry {}] Backing off for {} seconds...".format(n, \
                current_backoff))
            time.sleep(current_backoff)
        