from PIL import Image
from tqdm import tqdm
from operator import itemgetter
import torch
import random
from tqdm import tqdm


def pil_images_from_h5py(array):
    """
    Reading h5py file gives ndarry of shape (#elements, H, W)
    Function outputs a list of PIL images
    """
    train_cxr_pil = []
    for arr in tqdm(array):
        train_cxr_pil.append(Image.fromarray(arr).convert('RGB'))
    return train_cxr_pil

def tokenize_reports(wrapped_tokenizer, reports):
    """
    Tokenize the report dataset's section
    """
    return [wrapped_tokenizer(x,padding='max_length', max_length=33, truncation=True).input_ids for x in tqdm(reports)]

def intersection(lst1, lst2):
    """
    Implementing simple intersection between two lists
    """
    return list(set(lst1) & set(lst2))

def couple_most_different_elems(reports):
    """
    In order to implement Contrastive Learning we need to couple elements who are most different between each other.
    """
    indexes = []
    for y_idx in tqdm(range(len(reports))):
        min_intersection_value = 1 
        idx = 0
        for i in range(len(reports)):
            percentage = len(intersection(reports[y_idx],reports[i]))/len(list(set(reports[y_idx])))
            if percentage < min_intersection_value and (i not in indexes):
                min_intersection_value = percentage
                idx = i
        indexes.append(idx)
    return indexes

def generate_random_couple(reports):
    indexes = []
    for y_idx in range(len(reports)):
        x = random.randint(0,len(reports))
        if x == y_idx:
            x = y_idx + 1
        indexes.append(x)
    return indexes

def beam_search(p, k, beam, gpt):
    """
    Beam search algorithm with the goal of generating the most probable sequence of tokens.
    """
    candidates = []
    top_k = torch.topk(p[0], k)
    for i in range(len(top_k.indices)):
        candidates.append((top_k.values[i].item(), [top_k.indices[i].item()]))
    candidates = candidates[:beam]
    for i in range(1, len(p)):
        top_k = torch.topk(p[i], k)
        n_cands = []
        for v,idx in candidates:
            probs = gpt(torch.Tensor(idx).unsqueeze(0).long().cuda())
            top_k_probs = torch.index_select(probs[0],-1,torch.Tensor(top_k.indices))
            sum_probs = top_k_probs[0][-1]+ v
            for i in range(len(top_k.indices)):
                n_cands.append((sum_probs[i].item(), idx + [top_k.indices[i].item()]))
        s = sorted(n_cands, key=itemgetter(0))
        candidates = s[-beam:]
    return candidates


class EarlyStopping():
	def __init__(self, patience, delta=0):
		self.patience = patience 
		self.delta = delta
		self.count = 0
		self.loss = float('inf')

	def stop(self, loss):
		if self.loss == float('inf'):
			self.loss = loss
			return False
		if abs((loss - self.loss)) < self.delta and self.count >= self.patience:
			return True
	
		if abs((loss - self.loss)) < self.delta and self.count < self.patience:
			self.count +=1
			self.loss = loss
		else:
			self.loss = loss
			self.count = 0 
		return False
