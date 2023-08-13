import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, DataCollatorForTokenClassification

from spkatt_gepade import SPKATT_GEPADE_LABELS
from spkatt_gepade.bert_for_multi_label_token_classification import BertForMultiLabelTokenClassification
from spkatt_gepade.input_tokenizers import gen_role_sequence


def compute_metrics(true_labels, predictions_logits, **kwargs):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions_logits) > 0

    pred_labels = [x for t, x in zip(true_labels, predictions) if all(t != -100)]
    true_labels = [x for x in true_labels if all(x != -100)]

    print(classification_report(y_pred=pred_labels, y_true=true_labels, **kwargs))
    return f1_score(y_pred=pred_labels, y_true=true_labels, average='micro')


num_labels = len(SPKATT_GEPADE_LABELS)
label2id = {v: k for k, v in enumerate(SPKATT_GEPADE_LABELS)}
id2label = {k: v for k, v in enumerate(SPKATT_GEPADE_LABELS)}

BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME', 'aehrm/gepabert')
TRAIN_FILES = os.getenv('TRAIN_FILES', './data/train/task1')
DEV_FILES = os.getenv('DEV_FILES', './data/dev/task1')
MODEL_OUTPUT_DIR = str(Path(os.getenv('PEFT_MODEL_DIR', 'models')) / 'role_model_peft')

tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
tokenizer.add_special_tokens({'additional_special_tokens': ['[LABEL]']})
collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=[-100] * num_labels, padding=True)

input_seqs_train, input_seqs_dev = [], []
print('preparing input sequences: train')
for fname in tqdm(list(Path(TRAIN_FILES).glob('*.json'))):
    obj = json.load(open(fname))
    input_seqs_train.extend(
        gen_role_sequence(tokenizer, sentence_objects=obj['Sentences'], annotation_objects=obj['Annotations'],
                          add_labels=True, label_array=SPKATT_GEPADE_LABELS))

print('preparing input sequences: dev')
for fname in tqdm(list(Path(DEV_FILES).glob('*.json'))):
    obj = json.load(open(fname))
    input_seqs_dev.extend(
        gen_role_sequence(tokenizer, sentence_objects=obj['Sentences'], annotation_objects=obj['Annotations'],
                          add_labels=True, label_array=SPKATT_GEPADE_LABELS))


print('loading model')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = BertForMultiLabelTokenClassification.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels).to(device)
base_model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# IMPORTANT: add token embeddings to modules that need to be trained/saved, since we resized the embeddings in the
# base model; also add the pooler since it is initialized randomly
lora_config.modules_to_save = ['classifier', 'score', '.embeddings']

model = get_peft_model(base_model, lora_config)
# NOTE: we still disable training of the input embedding, and deal with the randomly initialized vector for [LABEL]
# (just store, not train!)
for n, p in model.named_parameters():
    if '.embeddings' in n:
        p.requires_grad = False

model.print_trainable_parameters()

print('starting training')
num_epochs = 30
batch_size = 4
gradient_accum = 1
lr = 5e-5

optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

train_loader = DataLoader(input_seqs_train, collate_fn=collator, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(input_seqs_dev, collate_fn=collator, batch_size=16)

best_f1_score = 0.0
best_model_state_dict = None

model = model.to(device)
progress_bar = tqdm(total=num_epochs * len(train_loader), unit="batch", ncols=80)
for epoch in range(num_epochs):
    progress_bar.set_description(f'epoch {epoch + 1}/{num_epochs}')
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        outputs = model(**batch.to(device), ignore_index=-100)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if i % gradient_accum == gradient_accum - 1 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
        progress_bar.set_postfix({'loss': loss.item()})
        progress_bar.update(1)

    avg_loss = total_loss / len(train_loader)
    model.eval()
    true_labels = []
    pred_logits = []
    for batch in dev_loader:
        with torch.no_grad():
            outputs = model(**batch.to(device))

        true_labels.extend(batch['labels'].cpu().reshape(-1, model.config.num_labels).numpy().tolist())
        pred_logits.extend(outputs.logits.cpu().reshape(-1, model.config.num_labels).numpy().tolist())

    score = compute_metrics(true_labels, pred_logits, target_names=SPKATT_GEPADE_LABELS)
    if score > best_f1_score:
        best_f1_score = score
        best_model_state_dict = copy.deepcopy(model.state_dict())

    old_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{num_epochs} | LR: {old_lr:.4e} | Avg Loss: {avg_loss:.4g} | F1 on dev: {score}")

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
