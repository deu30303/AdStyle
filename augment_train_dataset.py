import openai
import json
from transformers import AutoTokenizer
openai.api_key = 'API_KEY'

with open('./dataset/politifact_train_data.jsonl', "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
    
    
tone_list = ["objective and professional", "neutral", "emotionally triggering", "sensational"]
idx = 0
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

for dd in data[4800:]:
    print(idx)
    for tone in tone_list: 
        prompt = 'Rewrite the following article in {} tone:'.format(tone)
        input_text = dd['text']
        if len(tokenizer.encode(input_text)) > 4096:
            input_text = tokenizer.decode(tokenizer.encode(input_text[:4096]))
        inputs = f"{prompt} \n {input_text}"
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role":"user", "content":inputs}],
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        n=1,
        )
        paraphrase_outputs = response["choices"][0]["message"]["content"]
        dd[tone] = paraphrase_outputs
    idx += 1
    
    with open('./dataset/politifact_train_data.jsonl_pp', "a") as f:
        f.write(json.dumps(dd) + "\n")
    

    