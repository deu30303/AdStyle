import openai
import json
import transformers
from transformers import AutoTokenizer
import time 

openai.api_key = 'API_KEY'

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

with open('./dataset/politifact_test_data.jsonl', "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
    
    
publiser_list = ["CNN", "The New York Times", "National Enquirer", "The Sun"]
idx = 0
max_try_num = 10


for dd in data:
    print(idx)
    for tone in publiser_list: 
        prompt = 'Rewrite the following article as the writing style of {}:'.format(tone)
        curr_try_num = 0
        input_text = dd['text']
        
        if len(tokenizer.encode(input_text)) > 4096:
            input_text = tokenizer.decode(tokenizer.encode(input_text[:4096]))
            
        inputs = f"{prompt} \n {input_text}"

        while curr_try_num < max_try_num:
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role":"user", "content":inputs}],
                temperature=0.0,
                max_tokens=512,
                top_p=1,
                n=1,
                )
                paraphrase_outputs = response["choices"][0]["message"]["content"]
                break
                
            except Exception as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
        print("original outputs : ", input_text)
        print(tone, paraphrase_outputs)
        dd[tone] = paraphrase_outputs
        idx += 1
        
    with open('./dataset/politifact_test_data.jsonl_pp', "a") as f:
        f.write(json.dumps(dd) + "\n")

    