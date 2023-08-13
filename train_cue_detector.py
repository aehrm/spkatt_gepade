import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, BertForTokenClassification

from spkatt_gepade.input_tokenizers import get_cue_sequence

warnings.simplefilter("ignore", UndefinedMetricWarning)


def compute_metrics(true_labels, predictions_logits, **kwargs):
    true_labels = np.minimum(np.array(true_labels), 1)
    predictions = np.argmax(np.array(predictions_logits), axis=1)

    pred_labels = [x for t, x in zip(true_labels, predictions) if t != -100]
    true_labels = [t for t in true_labels if t != -100]

    print(classification_report(y_pred=pred_labels, y_true=true_labels, **kwargs))
    return f1_score(y_pred=pred_labels, y_true=true_labels, average='binary')


model_name = os.getenv('MODEL_PATH', 'aehrm/gepabert')
TRAIN_FILES = os.getenv('TRAIN_FILES', './data/train/task1')
DEV_FILES = os.getenv('DEV_FILES', './data/dev/task1')
MODEL_OUTPUT_DIR = str(Path(os.getenv('MODEL_OUTPUT_DIR', 'models')) / 'cue_model_peft')

tokenizer = BertTokenizerFast.from_pretrained(model_name)
collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100, padding=True)

input_seqs_train, input_seqs_dev = [], []
print('preparing input sequences: train')
for fname in tqdm(list(Path(TRAIN_FILES).glob('*.json'))):
    obj = json.load(open(fname))
    input_seqs_train.extend(
        get_cue_sequence(tokenizer, sentence_objects=obj['Sentences'], annotation_objects=obj['Annotations'],
                         add_labels=True))

print('preparing input sequences: dev')
for fname in tqdm(list(Path(DEV_FILES).glob('*.json'))):
    obj = json.load(open(fname))
    input_seqs_dev.extend(
        get_cue_sequence(tokenizer, sentence_objects=obj['Sentences'], annotation_objects=obj['Annotations'],
                         add_labels=True))

print('loading model')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = BertForTokenClassification.from_pretrained(model_name,
                                                        num_labels=2, id2label={0: 'O', 1: 'CUE'},
                                                        label2id={'O': 0, 'CUE': 1}
                                                        ).to(device)
lora_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

print('starting training')
num_epochs = 30
batch_size = 4
gradient_accum = 1
lr = 5e-5
train_loader = DataLoader(input_seqs_train, collate_fn=collator, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(input_seqs_dev, collate_fn=collator, batch_size=16)

optimizer = AdamW(params=model.parameters(), lr=lr, eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

best_f1_score = 0.0
best_model_state_dict = None

progress_bar = tqdm(total=num_epochs * len(train_loader), unit="batch", ncols=80)
for epoch in range(num_epochs):
    progress_bar.set_description(f'epoch {epoch + 1}/{num_epochs}')
    model.train()

    true_token_labels = []
    pred_token_logits = []
    total_loss = 0
    for i, batch in enumerate(train_loader):
        output = model(**batch.to(device))
        loss = output.loss
        total_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if i % gradient_accum == gradient_accum - 1 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
        progress_bar.set_postfix({'loss': loss.item()})
        progress_bar.update(1)

        labels = batch['labels'].detach().cpu()
        true_token_labels.extend(labels.reshape(-1).numpy().tolist())
        pred_token_logits.extend(output.logits.detach().cpu().reshape(-1, 2).numpy().tolist())

    print('metrics on train')
    compute_metrics(true_token_labels, pred_token_logits, target_names=['O', 'CUE'])

    model.eval()
    true_token_labels = []
    pred_token_logits = []
    true_pair_labels = []
    pred_pair_logits = []
    for batch in dev_loader:
        labels = batch['labels'].detach().cpu()
        with torch.no_grad():
            output = model(**batch.to(device))

        true_token_labels.extend(labels.reshape(-1).numpy().tolist())
        pred_token_logits.extend(output.logits.detach().cpu().reshape(-1, 2).numpy().tolist())

    print('metrics on dev')
    score = compute_metrics(true_token_labels, pred_token_logits, target_names=['O', 'CUE'])

    if score > best_f1_score:
        best_f1_score = score
        best_model_state_dict = copy.deepcopy(model.state_dict())

    old_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch + 1}/{num_epochs} | LR: {old_lr:.4e} | Avg Loss: {total_loss / len(train_loader):.4g} | F1 on dev: {score:.2f}")

    scheduler.step(score)
    new_lr = optimizer.param_groups[0]['lr']

    if new_lr <= scheduler.min_lrs[0]:
        print('LR too low, finishing')
        break
    if old_lr != new_lr:
        print('adjusted lr, restoring best model')
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)

print(f'loading model with best score on dev: {best_f1_score}, saving to {MODEL_OUTPUT_DIR}')
model.load_state_dict(best_model_state_dict)
model.save_pretrained(MODEL_OUTPUT_DIR)
