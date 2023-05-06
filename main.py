# %% [code]
from transformers import GPT2Tokenizer, GPT2LMHeadModel
MODEL_NAME = 'distilgpt2' 
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)


# %% [code]
# Declare special tokens for padding and separating the movie_plot from the movie_title:
SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<movie_plot>', '<movie_title>'],
}

# Add these special tokens to the vocabulary and resize model's embeddings:
tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
model.resize_token_embeddings(len(tokenizer))

# Show the full list of special tokens:
print(tokenizer.special_tokens_map)


# %% [code]
import csv

import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
  def __init__(self, filename, tokenizer, seq_length=64):

    movie_plot_tkn = tokenizer.additional_special_tokens_ids[0]
    movie_title_tkn = tokenizer.additional_special_tokens_ids[1]
    pad_tkn = tokenizer.pad_token_id
    eos_tkn = tokenizer.eos_token_id

    self.examples = []
    with open(filename) as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
      
        # Build the movie_plot and movie_title segments:
        movie_plot = [movie_plot_tkn] + tokenizer.encode(row[0], max_length=seq_length//2-1)
        movie_title = [movie_title_tkn] + tokenizer.encode(row[1], max_length=seq_length//2-2) + [eos_tkn]
        
        # Concatenate the two parts together:
        tokens = movie_plot + movie_title + [pad_tkn] * ( seq_length - len(movie_plot) - len(movie_title) )

        # Annotate each token with its corresponding segment:
        segments = [movie_plot_tkn] * len(movie_plot) + [movie_title_tkn] * ( seq_length - len(movie_plot) )

        # Ignore the movie_plot, padding, and <movie_title> tokens by setting their labels to -100
        labels = [-100] * (len(movie_plot)+1) + movie_title[1:] + [-100] * ( seq_length - len(movie_plot) - len(movie_title) )

        # Add the preprocessed example to the dataset
        self.examples.append((tokens, segments, labels))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, item):
    return torch.tensor(self.examples[item])


# Build the dataset and display the dimensions of the 1st batch for verification:
movie_title_dataset = MovieDataset('movie_titles.csv', tokenizer)
print(next(iter(movie_title_dataset)).size())

# %% [code]
import math, random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# Create data indices for training and validation splits:

indices = list(range(len(movie_title_dataset)))

random.seed(42)
random.shuffle(indices)

split = math.floor(0.1 * len(movie_title_dataset))
train_indices, val_indices = indices[split:], indices[:split]

# Build the PyTorch data loaders:

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(movie_title_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(movie_title_dataset, batch_size=64, sampler=val_sampler)



# %% [code]
import numpy as np
from tqdm import tqdm


def fit(model, optimizer, train_dl, val_dl, epochs=1, device=torch.device('cpu')):

  for i in range(epochs):

    print('\n--- Starting epoch #{} ---'.format(i))

    model.train()

    # These 2 lists will keep track of the batch losses and batch sizes over one epoch:
    losses = []
    nums = []

    for xb in tqdm(train_dl, desc="Training"):
      # Move the batch to the training device:
      inputs = xb.to(device)

      # Call the model with the token ids, segment ids, and the ground truth (labels)
      outputs = model(inputs[:,0,:], token_type_ids=inputs[:,1,:], labels=inputs[:,2,:])
      
      # Add the loss and batch size to the list:
      loss = outputs[0]
      losses.append(loss.item())
      nums.append(len(xb))

      loss.backward()

      optimizer.step()
      model.zero_grad()

    # Compute the average cost over one epoch:
    train_cost = np.sum(np.multiply(losses, nums)) / sum(nums)


    # Now do the same thing for validation:

    model.eval()
    
    with torch.no_grad():
      losses = []
      nums = []

      for xb in tqdm(val_dl, desc="Validation"):
        inputs = xb.to(device)
        outputs = model(inputs[:,0,:], token_type_ids=inputs[:,1,:], labels=inputs[:,2,:])
        losses.append(outputs[0].item())
        nums.append(len(xb))

    val_cost = np.sum(np.multiply(losses, nums)) / sum(nums)

    print('\n--- Epoch #{} finished --- Training cost: {} / Validation cost: {}'.format(i, train_cost, val_cost))


# %% [code]
from transformers import AdamW

# Move the model to the GPU:
device = torch.device('cpu')
model.to(device)

# Fine-tune GPT2 for two epochs:
optimizer = AdamW(model.parameters())
fit(model, optimizer, train_loader, val_loader, epochs=2, device=device)


# %% [code]
# Sampling functions with top k and top p from HuggingFace:

import torch.nn.functional as F
from tqdm import trange


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


# From HuggingFace, adapted to work with the movie_plot/movie_title separation:
def sample_sequence(model, length, movie_plot, segments_tokens=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu'):
    movie_plot = torch.tensor(movie_plot, dtype=torch.long, device=device)
    movie_plot = movie_plot.unsqueeze(0).repeat(num_samples, 1)
    generated = movie_plot

    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if segments_tokens != None:
              inputs['token_type_ids'] = torch.tensor(segments_tokens[:generated.shape[1]]).unsqueeze(0).repeat(num_samples, 1)


            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


# %% [code]
movie_plot = "Starbucks, coffee chain from Seattle"

movie_plot_tkn = tokenizer.additional_special_tokens_ids[0]
movie_title_tkn = tokenizer.additional_special_tokens_ids[1]

input_ids = [movie_plot_tkn] + tokenizer.encode(movie_plot)

segments = [movie_title_tkn] * 64
segments[:len(input_ids)] = [movie_plot_tkn] * len(input_ids)

input_ids += [movie_title_tkn]

# Move the model back to the CPU for inference:
model.to(torch.device('cpu'))

# Generate 20 samples of max length 20
generated = sample_sequence(model, length=20, movie_plot=input_ids, segments_tokens=segments, num_samples=20)

print('\n\n--- Generated movie titles from your plot ---\n')

for g in generated:
  movie_title = tokenizer.decode(g.squeeze().tolist())
  movie_title = movie_title.split('<|endoftext|>')[0].split('<movie_title>')[1]
  print(movie_title)
\end{python}

