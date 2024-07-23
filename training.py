import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup,  LlamaForCausalLM, LlamaTokenizer
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import macros
import utilities
import random
import pandas as pd 
import h5py
import datetime
import time
import numpy as np
from transformers import BioGptTokenizer

MODEL_TYPE = "GPT"

# Preprocess the data

train_reports = []
with open('file2.txt') as f:
    data = f.read()
    train_reports = data.split('\n')
train_reports = train_reports[:-1]

test_reports = pd.read_csv(macros.TEST_REPORTS_PATH)['report']
train_cxr = h5py.File(macros.TRAIN_CXR_FILE , "r")
test_cxr = h5py.File(macros.TEST_CXR_FILE , "r")

train_cxr_pil = utilities.pil_images_from_h5py(train_cxr['cxr'])
test_cxr_pil = utilities.pil_images_from_h5py(test_cxr['cxr'])

import re
resampled_train_cxr = []
single_phrase_report = []
for i in tqdm(range(len(train_reports))):
    splitted = re.split(r'\d. |- ', train_reports[i])
    if len(splitted) > 1:
        for s in splitted:
            if s:
                single_phrase_report.append(s)
                resampled_train_cxr.append(train_cxr_pil[i])
    else:
        single_phrase_report.append(train_reports[i])
        resampled_train_cxr.append(train_cxr_pil[i])


tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT-Large')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

indices = []
for i in tqdm(range(len(single_phrase_report))):
    if not single_phrase_report[i]:
        indices.append(i)

def remove_from_list(input_list: list, indices: list):
    for i in sorted(indices, reverse=True):
        del input_list[i]

remove_from_list(single_phrase_report, indices)
remove_from_list(resampled_train_cxr, indices)

class GPT2Dataset(Dataset):

  def __init__(self, image_list, txt_list, tokenizer, processor, max_length=64):

    self.input_ids = []
    self.attn_masks = []
    self.pixel_values = []

    for txt in tqdm(txt_list):

      encodings_dict = tokenizer( txt, truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    for img in tqdm(image_list):
      self.pixel_values.append(processor(img, return_tensors="pt")['pixel_values'].squeeze())
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):

    image = self.pixel_values[idx]
    ids = self.input_ids[idx]
    label = torch.concat( (torch.tensor(-100).repeat(197), ids))
    attn_mask = torch.concat((torch.tensor(1).repeat(197), self.attn_masks[idx]))
    position_ids = torch.concat((torch.tensor(labels),torch.range(4,35)))
    token_type_ids = torch.concat((torch.zeros(197),torch.ones(32)))
    return ids, image, label, attn_mask, position_ids, token_type_ids

dataset = GPT2Dataset(resampled_train_cxr, single_phrase_report, tokenizer, processor,32)

# Split into training and validation sets
train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = 16,
            shuffle=True # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            batch_size = 8,
            shuffle=True # Evaluate with this batch size.
        )

class MultiModalTransformer(nn.Module):
    def __init__(self, language_model, image_processor):
        super().__init__()
        self.llm = language_model
        self.image_processor = image_processor
        #self.ln = nn.LayerNorm(768)
        if MODEL_TYPE == 'LLAMA':
            self.wte = language_model.model.embed_tokens
        else:
            self.wte = language_model.transformer.wte
        #self.gelu = nn.GELU()

    def preprocess_img(self, img):
        return self.image_processor(img)

    def forward(self, img, ids, lbl, attn_masks, position_ids,token_type_ids):
        img = self.preprocess_img(img)
        ids = self.wte(ids)
        input_embeds = torch.cat((img, ids), dim=1)
        out = self.llm.forward(inputs_embeds=input_embeds.cuda(), labels=lbl.long().cuda(), attention_mask=attn_masks, position_ids=position_ids, token_type_ids=token_type_ids, output_attentions=True)
        return out


vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

epochs = 15
learning_rate = 5e-4
warmup_steps = 10e2
epsilon = 1e-8
sample_every = 500
multimodal_model = MultiModalTransformer(model, vit.vit.embeddings)
model = multimodal_model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon,
                )
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)
# Greedy search
def generate(img_emb):
    softmax = nn.Softmax(dim = 1)
    t=model.preprocess_img(img_emb.unsqueeze(0))
    generated_ids = []
    i = 0
    for i in range(256):
        pred = model.llm(inputs_embeds=t)['logits']
        pred = softmax(pred)
        p = torch.topk(pred.squeeze()[-1],1)
        token = p.indices.item()
        if token == tokenizer.pad_token_id:
            break
        generated_ids.append(token)
        emb = model.wte(torch.tensor(token).cuda())
        t = torch.concat((t.squeeze(),emb.unsqueeze(0)),dim = 0)
    return generated_ids

# Beam search
from operator import itemgetter
import math
def beam_search(k, beam, gpt, img_emb):
    softmax = nn.Softmax(dim = 1)
    """
    Beam search algorithm with the goal of generating the most probable sequence of tokens.
    """
    t=model.preprocess_img(img_emb.unsqueeze(0))
    token_type_ids = torch.zeros(197).int().cuda().unsqueeze(0)
    position_ids = torch.zeros(197).int().cuda().unsqueeze(0)
    pred = model.llm(inputs_embeds=t, position_ids=position_ids, token_type_ids=token_type_ids)['logits'].squeeze()
    
    pred = softmax(pred)
    top_k = torch.topk(pred[-1],k)
    candidates = []
    for i in range(len(top_k.indices)):
        candidates.append((top_k.values[i].item(), [top_k.indices[i].item()]))
    candidates = candidates[:beam]
    for i in range(32):
        token_type_ids = torch.cat((token_type_ids, torch.ones(1).int().cuda().unsqueeze(0)), dim=1)
        n_cands = []
        for v,idx in candidates:
            emb = model.wte(torch.tensor(idx).cuda())
            tt = torch.concat((t.squeeze(),emb),dim = 0)
            position_ids = torch.concat((torch.zeros(197),torch.range(1,i+1))).int().cuda().unsqueeze(0)
            pred = model.llm(inputs_embeds=tt, position_ids=position_ids, token_type_ids=token_type_ids)['logits']
            
            pred = softmax(pred)
            top_k = torch.topk(pred[-1],k)
            top_k_probs = top_k.values
            sum_probs = top_k_probs+ v
            for j in range(len(top_k.indices)):
                already_in_list = False
                for x,y in n_cands:
                   
                    if idx[0] in y:
                        already_in_list = True
                if not already_in_list and not top_k.indices[j].item() == tokenizer.pad_token_id:
                    
                    n_cands.append((sum_probs[j].item() * (1 - (32-(len(idx)+1))), idx + [top_k.indices[j].item()]))
        s = sorted(n_cands, key=itemgetter(0),reverse=True)
        candidates = s[:beam]
    return candidates

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


# Training loop
total_t0 = time.time()

training_stats = []


for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(macros.DEVICE)
        b_imgs = batch[1].to(macros.DEVICE)
        b_labels = batch[2].to(macros.DEVICE)
        b_attn_mask = batch[3].to(macros.DEVICE)
        b_position_ids = batch[4].to(macros.DEVICE)
        b_token_type_ids = batch[5].to(macros.DEVICE)
   
        model.zero_grad()        

        outputs = model(  b_imgs,
                          b_input_ids,
                          b_labels,
                          b_attn_mask,
                          b_position_ids.int(),
                          b_token_type_ids.long()
                        )
        loss = outputs[0]  

        batch_loss = loss.mean().item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()
            print(f'Real: {tokenizer.decode(b_labels[0][197:])}')
            bs_out = beam_search_decoding(50,10,b_imgs[0],32)
            print(f'Greedy: {tokenizer.decode(generate(b_imgs[0]))}')
            for score,generated_ids in bs_out: 
                print(f'{score} {tokenizer.decode(generated_ids)}')
            
            model.train()

        loss.mean().backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(macros.DEVICE)
        b_imgs = batch[1].to(macros.DEVICE)
        b_labels = batch[2].to(macros.DEVICE)
        b_attn_mask = batch[3].to(macros.DEVICE)
        b_position_ids = batch[4].to(macros.DEVICE)
        b_token_type_ids = batch[5].to(macros.DEVICE)
   

        with torch.no_grad():        

            outputs = model(  b_imgs,
                          b_input_ids,
                          b_labels,
                          b_attn_mask,
                          b_position_ids.int(),
                          b_token_type_ids.long()
                        )
            loss = outputs[0] 
          
            
        batch_loss = loss.mean().item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        