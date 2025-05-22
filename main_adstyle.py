import argparse
import json
import tqdm
import functools
import os
import numpy as np
import pickle
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AdamW
import torch.nn.functional as F
import openai
import time
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default="./dataset/politifact_train_data.jsonl_pp")
    parser.add_argument('--test_dir', default="./dataset/politifact_test_data.jsonl_pp")
    parser.add_argument('--pretrained_dir', default="./models/politifact_augment_detector.pt")
    parser.add_argument('--api_key', default="API_KEY")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--round', default=10, type=int)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--max_len', default=512, type=int)
    
    args = parser.parse_args()
    
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    
class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.l1.classifier = nn.Sequential()
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids, embed=False):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        hidden_state = output_1[0]
        embedding = hidden_state[:, 0]
        
        pooler = self.pre_classifier(embedding)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        
        if embed == False:
            return output
        else:
            return output, embedding
        
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# initialization algorithm
def initialize_centroid(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters 
    '''
    # initialize the centroids list and add
    # a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0]), :])

    # compute remaining k - 1 centroids
    for c_id in range(k - 1):

        # initialize a list to store distances of data
        # points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            # compute distance of 'point' from each of the previously
            # selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        
    return centroids


def compare_semantic(original, paraphrase, api_key):
    max_try_num=10 
    prompt = "Given two pieces of text, Original and Paraphrase, determine whether the two sentences are semantically similar. Answer Yes or No only."
    
    text = "Original: {} \n\n Paraphrase: {}\n\n".format(original,paraphrase)
    
    
    inputs = f"{prompt}: \n {text}"

    curr_try_num = 0
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
            paraphrase_text = response["choices"][0]["message"]["content"]                
            break
        except openai.error.InvalidRequestError as e:
            return 0
        except Exception as e:
            print(e)
            curr_try_num += 1
            if curr_try_num >= max_try_num:
                result_list.append(-1)
            time.sleep(10)
            
    if "Yes" in response["choices"][0]["message"]["content"]:
        score = 1
    else:
        score = 0
        
    return score



def evaluate_prompt(prompt, eval_data, model, tokenizer, api_key):
    max_try_num=10 
    openai.api_key = api_key
    label_list = []
    prediction_list = []
    embedding_list = []
    result_list = []
    sim_result = []
    

    model.eval()
    for data in eval_data: 
        oringal_text = data['text']
        label = data['label'].split()[0]
        
        if label == 'fake':
            label_list.append(0)
        elif label == 'real':
            label_list.append(1)
                
        if prompt in  data.keys():
            paraphrase_text = data[prompt]
            
        else:
            if len(tokenizer.encode(oringal_text)) > 512:
                oringal_text = tokenizer.decode(tokenizer.encode(oringal_text[:512]))

            inputs = f"{prompt}: \n {oringal_text}"

            curr_try_num = 0
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
                    paraphrase_text = response["choices"][0]["message"]["content"]                
                    break
                except openai.error.InvalidRequestError as e:
                    return [-1]
                except Exception as e:
                    print(e)
                    curr_try_num += 1
                    if curr_try_num >= max_try_num:
                        result_list.append(-1)
                    time.sleep(10)

            data[prompt] = paraphrase_text
            
            
        news_input = tokenizer.encode_plus(paraphrase_text,  None, add_special_tokens=True, max_length=512, \
                                                  pad_to_max_length=True, return_token_type_ids=False)
        news_input_ids = torch.tensor(news_input['input_ids']).unsqueeze(0).cuda()
        news_attention_mask = torch.tensor(news_input['attention_mask']).unsqueeze(0).cuda()
        
        output, embedding = model(news_input_ids, news_attention_mask, None, embed=True)
        embedding_list.append(embedding.cpu().detach())
        prediction_list.append(output.item())
        
        sim_score = compare_semantic(oringal_text, paraphrase_text, api_key)
        
        sim_result.append(sim_score)
        
        del embedding, output, news_input, news_input_ids, news_attention_mask
        
    entire_embedding = torch.cat(embedding_list)    
    promt_embed = torch.mean(entire_embedding, dim=0) 
    
    similarity_mean = max(0.1, np.mean(sim_result))
    
    
    auroc = roc_auc_score(np.array(label_list), np.array(prediction_list))
    
    return auroc, eval_data, promt_embed, similarity_mean


def generate_promt(prompt_value_dict, api_key):
    max_try_num=10 
    openai.api_key = api_key
    result_list = []
    
    sorted_dict = sorted(prompt_value_dict.items(), key= lambda item:item[1], reverse=True)
    
    prompt = "Now you will help me minimize a fake news detector performance value with style transfer attack instruction. I have some style transfer attack instructions and the fake news detector performance value using those instructions. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n"
    
    for key, value in sorted_dict:
        element = "input: %s \nvalue: %.2f\n\n"%(key, value)
        prompt += element
        
    prompt+= "\n\nGive me a new style transfer attack instruction that is different from all pairs above, and has a performance value lower than any of the above.\n"

    curr_try_num = 0
    
    while curr_try_num < max_try_num:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role":"user", "content":prompt}],
            temperature=0.0,
            max_tokens=512,
            top_p=1,
            n=1,
            )
            answer = response["choices"][0]["message"]["content"]
            generated_prompt = answer.split("\n")[0].split(":")[1]
            break
        except openai.error.InvalidRequestError as e:
            return [-1]
        except Exception as e:
            print(e)
            curr_try_num += 1
            if curr_try_num >= max_try_num:
                result_list.append(-1)
            time.sleep(10)
            
    
    return generated_prompt.strip()

def augment_train_dataset(prompt, train_data, api_key):
    max_try_num=10 
    openai.api_key = api_key
    result_list = []
    
    print("AUGMENT: ", prompt)
    
    for data in train_data:
        try:
            paraphrase_text = data[prompt]
        except:
            oringal_text = data['text']
            label = data['label'].split()[0]

            if len(tokenizer.encode(oringal_text)) > 512:
                oringal_text = tokenizer.decode(tokenizer.encode(oringal_text[:512]))

            inputs = f"{prompt}: \n {oringal_text}"

            curr_try_num = 0

            while curr_try_num < max_try_num:
                try:
                    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[{"role":"user", "content":inputs}],
                    temperature=0.7,
                    max_tokens=512,
                    top_p=1,
                    n=1,
                    )
                    paraphrase_text = response["choices"][0]["message"]["content"]                
                    break
                except openai.error.InvalidRequestError as e:
                    return [-1]
                except Exception as e:
                    print(e)
                    curr_try_num += 1
                    if curr_try_num >= max_try_num:
                        result_list.append(-1)
                    time.sleep(10)
                    
        data[prompt] = paraphrase_text
                
    return train_data
        

    
class FakeNewsData(Dataset):
    def __init__(self, data, tokenizer, max_len, train, prompt_list=[]):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.newstext_list = []
        self.label_list = []
        
        for index in range(0, len(self.data)):
            if train == True:
                tone_list = ["text"] + prompt_list
                news_inputs = {}
                label = self.data[index]['label'].split()[0]

                if label == 'fake':
                    self.label_list.append(0)
                elif label == 'real':
                    self.label_list.append(1)
                    
                news_input_ids = torch.tensor([])
                news_attention_mask = torch.tensor([])
                    
                for tone in tone_list:
                    news_tokens = self.data[index][tone].split()
                    news_text = " ".join(news_tokens)     
                    news_input = tokenizer.encode_plus(news_text,  None, add_special_tokens=True, max_length=self.max_len, \
                                                  pad_to_max_length=True, return_token_type_ids=False)

                    news_input_ids = torch.cat((torch.tensor(news_input['input_ids']), news_input_ids))
                    news_attention_mask = torch.cat((torch.tensor(news_input['attention_mask']), news_attention_mask))
                self.newstext_list.append([news_input_ids, news_attention_mask])
                
            elif train == False:
                publiser_list = ["CNN", "The New York Times", "The Sun", "National Enquirer" ]
                news_inputs = {}
                label = self.data[index]['label'].split()[0]

                if label == 'fake':
                    self.label_list.append(0)
                elif label == 'real':
                    self.label_list.append(1)
                    
                news_input_ids = torch.tensor([])
                news_attention_mask = torch.tensor([])
                    
                for publiser in publiser_list:
                    news_tokens = self.data[index][publiser].split()
                    news_text = " ".join(news_tokens)     
                    news_input = tokenizer.encode_plus(news_text,  None, add_special_tokens=True, max_length=self.max_len, \
                                                      pad_to_max_length=True, return_token_type_ids=False)   
                    
                    news_input_ids = torch.cat((torch.tensor(news_input['input_ids']), news_input_ids))
                    news_attention_mask = torch.cat((torch.tensor(news_input['attention_mask']), news_attention_mask))
                self.newstext_list.append([news_input_ids, news_attention_mask])
                
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):                       
        news_input = self.newstext_list[index]
        news_label = self.label_list[index]

        return news_input[0], news_input[1], news_label
    

if __name__ == "__main__":
    args = get_args()
    set_seed(0)
    api_key = args.api_key
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    model = Detector()
    model.cuda()
    model.load_state_dict(torch.load(args.pretrained_dir))

    with open(args.train_dir, "r") as f:
        train_data = [json.loads(x) for x in f.read().strip().split("\n")]
        
    eval_data = random.sample(train_data, 30)
    
    with open(args.test_dir, "r") as f:
        test_data = [json.loads(x) for x in f.read().strip().split("\n")]
        
    print(len(train_data), len(test_data))
    testing_set = FakeNewsData(test_data, tokenizer, args.max_len, False)
    testing_loader = DataLoader(testing_set, batch_size =args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    default_prompt =['Rewrite the following article in objective and professional tone', 'Rewrite the following article in neutral tone', 'Rewrite the following article in emotionally triggering tone', 'Rewrite the following article in sensational tone']
    

    for rounds in range(0, args.round):
        print("Round : {}".format(rounds))
        prompt_value_dict = {}
        new_prompt_value_dict = {}
        subtract_list = []
        optimizer = AdamW(model.parameters(), lr = args.lr,  eps = 1e-8 )
        
        for promt in default_prompt:
            auroc, eval_data, promt_embed, _ = evaluate_prompt(promt, eval_data, model, tokenizer, api_key)
            prompt_value_dict[promt] = np.abs(auroc - 0.5).item()
     
        prompt_list = []
        _, _, original_embedd, _ = evaluate_prompt('text', eval_data, model, tokenizer, api_key)
            
        for i in range(0, 30):
            new_prompt = generate_promt(prompt_value_dict, api_key)
            auroc, eval_data, promt_embed, sim_val = evaluate_prompt(new_prompt, eval_data, model, tokenizer, api_key)
            uncertainty = -1.8*torch.abs(torch.tensor(auroc) - 0.5) + 1.0
            subtract_list.append((sim_val*uncertainty*(promt_embed - original_embedd)).unsqueeze(0))
            prompt_value_dict[new_prompt] = np.abs(auroc - 0.5)
            new_prompt_value_dict[new_prompt] = np.abs(auroc - 0.5)
            prompt_list.append(new_prompt)
             
        subtract_vectors = torch.cat(subtract_list, dim=0).numpy()
        centroids_list = initialize_centroid(subtract_vectors, 3)
        min_idx_list = []
        
        for centrioid in centroids_list:
            centrioid_ = torch.tensor(centrioid).unsqueeze(0)
            dist_ = torch.sqrt(torch.sum((centrioid_ -  torch.tensor(subtract_vectors))**2, dim=1))
            _, min_idx = torch.topk(-dist_, k=1)
            min_idx_list.append(min_idx.item())
        
        adv_prompt_list = []
        for idx in min_idx_list:   
            adv_prompt_list.append(prompt_list[idx])

        default_prompt += adv_prompt_list

        print("Selected Prompt : ", adv_prompt_list)
        print("Default Prompt : ", default_prompt)
        
        for min_prompt in adv_prompt_list:
            train_data = augment_train_dataset(min_prompt, train_data, api_key)
            
        training_set = FakeNewsData(train_data, tokenizer, args.max_len, True, adv_prompt_list)
        training_loader = DataLoader(training_set, batch_size =args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        model.train()
        bce_criterion = nn.BCELoss()

        for idx, data in enumerate(training_loader):
            batch_size = data[2].shape[0]
            news_inputs = data[0].cuda().long()
            print(news_inputs.shape)
            news_attention_mask = data[1].cuda().long()
            news_target = data[2].cuda().unsqueeze(1).float()
            

            news_inputs_list = news_inputs.split(args.max_len, dim=1)
            news_attention_mask_list = news_attention_mask.split(args.max_len, dim=1)

            news_inputs_cat = torch.cat(news_inputs_list, dim=0)
            news_attention_mask_cat = torch.cat(news_attention_mask_list, dim=0)

            news_target_cat = torch.cat([news_target]*len(news_inputs_list))


            model.zero_grad()
            output = model(news_inputs_cat, news_attention_mask_cat, None)
            loss = bce_criterion(output, news_target_cat) 

            loss.backward()
            optimizer.step()
            
        model_dir_name = './models/adstyle_round{}.pt'.format(rounds)
        torch.save(model.state_dict(), model_dir_name)

        model.eval()

        publiser_list = [ "CNN", "The New York Times", "The Sun", "National Enquirer" ]
        output_dict = {"CNN": [], "The New York Times": [], "The Sun": [], "National Enquirer":[] }

        target_list = []

        for idx, data in enumerate(testing_loader):
            batch_size = data[2].shape[0]
            news_inputs = data[0].cuda().long()


            news_attention_mask = data[1].cuda().long()
            news_target = data[2].cuda().unsqueeze(1)

            news_inputs_list = news_inputs.split(args.max_len, dim=1)
            news_attention_mask_list = news_attention_mask.split(args.max_len, dim=1)

            with torch.no_grad():
                for i in range(0, len(news_inputs_list)):
                    output = model(news_inputs_list[i],news_attention_mask_list[i], None)
                    output_dict[publiser_list[i]].append(output.detach().cpu())
            target_list.append(news_target.detach().cpu())

        target_ = torch.cat(target_list)
        for publiser in publiser_list: 
            output_ = torch.cat(output_dict[publiser]).squeeze(1)
            auroc1 = roc_auc_score(target_.numpy(), output_.numpy())
            print(publiser, " AUROC : ", auroc1)
