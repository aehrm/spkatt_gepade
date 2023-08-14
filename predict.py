import json
import os
import sys
from pathlib import Path

import more_itertools
import networkx as nx
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, BertForTokenClassification, \
    BertForSequenceClassification

from spkatt_gepade import SPKATT_GEPADE_LABELS
from spkatt_gepade.bert_for_multi_label_token_classification import BertForMultiLabelTokenClassification
from spkatt_gepade.common import matching_precision_recall_f1
from spkatt_gepade.input_tokenizers import get_cue_sequence, get_cue_link_sequence, gen_role_sequence

PEFT_MODEL_DIR = os.getenv('PEFT_MODEL_DIR', './models')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_input(list_of_paths):
    input_json = {}
    for f in list_of_paths:
        obj = json.load(open(f))
        input_json[f.name] = obj

    return input_json


def predict_cue_words(sentence_dict):
    peft_name = str(Path(PEFT_MODEL_DIR) / 'cue_model_peft')
    peft_config = PeftConfig.from_pretrained(peft_name)
    tokenizer = BertTokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    input_seqs = []
    for fname, obj in tqdm(sentence_dict.items(), desc='cue detection: tokenize'):
        input_seqs.extend(get_cue_sequence(tokenizer, sentence_objects=obj, fname=fname, return_coords=True))

    base_model = BertForTokenClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=2).to(
        device)
    model = PeftModel.from_pretrained(base_model, peft_name).eval()

    positive_coords = []
    with tqdm(desc='cue detection: predict', total=len(input_seqs)) as pbar:
        for batch in more_itertools.chunked(input_seqs, 16):
            batch, batch_token_coords, sentids = zip(*batch)
            batch_collated = collator(batch)
            with torch.no_grad():
                output = model(**batch_collated.to('cuda'))

            for logits, coords, focus_sent in zip(output.logits, batch_token_coords, sentids):
                for tok_logits, coord in zip(logits, coords):
                    if coord is None:
                        continue
                    _, sentid, _ = coord
                    if sentid != focus_sent:  # one sequence per sentence
                        continue
                    if tok_logits.argmax() == 1:
                        positive_coords.append(coord)
            pbar.update(len(batch))

    model.cpu()
    del model

    return list(sorted(positive_coords))


def predict_cue_links(sentence_dict, positive_coords):
    peft_name = str(Path(PEFT_MODEL_DIR) / 'cue_joiner_peft')
    peft_config = PeftConfig.from_pretrained(peft_name)

    tokenizer = BertTokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[LABEL]']})
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    input_seqs = []
    for fname, obj in tqdm(sentence_dict.items(), desc='cue links: tokenize'):
        positive_coords_in_f = [x for x in positive_coords if x[0] == fname]
        input_seqs.extend(
            get_cue_link_sequence(tokenizer, sentence_objects=obj, positive_cues=positive_coords_in_f, fname=fname,
                                  return_coords=True))

    base_model = BertForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=2).to(
        device)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, peft_name).eval()

    G = nx.Graph()
    G.add_nodes_from(positive_coords)

    with tqdm(desc='cue links: predict', total=len(input_seqs)) as pbar:
        for batch in more_itertools.chunked(input_seqs, 16):
            batch, pair_coords = zip(*batch)
            batch_collated = collator(batch)
            with torch.no_grad():
                output = model(**batch_collated.to('cuda'))

            for logits, (u, v) in zip(output.logits, pair_coords):
                if logits.argmax() == 1:
                    G.add_edge(u, v)

            pbar.update(len(batch))

    model.cpu()
    del model

    prediction_dict = {f: list() for f in sentence_dict.keys()}
    components = nx.connected_components(G)
    for comp in components:
        comp = list(comp)
        f, _, _ = comp[0]
        prediction_dict[f].append({'Cue': [f'{sentid}:{tokid}' for _, sentid, tokid in comp]})

    return prediction_dict


def predict_roles(sentence_dict, prediction_dict):
    new_prediction_dict = {}
    for fname, annotation_list in prediction_dict.items():
        new_prediction_dict[fname] = []
        for annot_obj in annotation_list:
            new_obj = {k: list() for k in SPKATT_GEPADE_LABELS}
            new_obj['Cue'] = annot_obj['Cue']

            new_prediction_dict[fname].append(new_obj)

    peft_name = str(Path(PEFT_MODEL_DIR) / 'role_model_peft')
    peft_config = PeftConfig.from_pretrained(peft_name)

    tokenizer = BertTokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[LABEL]']})
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    input_seqs = []
    for fname in tqdm(sentence_dict.keys(), desc='role detection: tokenize'):
        input_seqs.extend(gen_role_sequence(tokenizer, sentence_objects=sentence_dict[fname],
                                            annotation_objects=new_prediction_dict[fname], fname=fname,
                                            return_coords=True))

    base_model = BertForMultiLabelTokenClassification.from_pretrained(peft_config.base_model_name_or_path,
                                                                      num_labels=len(SPKATT_GEPADE_LABELS)).to(device)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, peft_name).eval()

    with tqdm(desc='role detection: predict', total=len(input_seqs)) as pbar:
        for batch in more_itertools.chunked(input_seqs, 16):
            batch, batch_token_coords, annotation_ids = zip(*batch)
            batch_collated = collator(batch)
            with torch.no_grad():
                output = model(**batch_collated.to('cuda'))

            for logits, coords, annotation_id in zip(output.logits, batch_token_coords, annotation_ids):
                for tok_logits, coord in zip(logits, coords):
                    if coord is None:
                        continue
                    for j, label in enumerate(SPKATT_GEPADE_LABELS):
                        if label == 'Cue':
                            continue
                        if tok_logits[j] > 0:
                            f, sentid, tokid = coord
                            new_prediction_dict[f][annotation_id][label].append(f'{sentid}:{tokid}')

            pbar.update(len(batch))

    model.cpu()
    del model

    return new_prediction_dict


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ['1a', '1b']:
        print('usage: predict.py 1a input_dir [output_dir]')
        print('    or predict.py 1b input_dir [output_dir]')
        sys.exit(1)

    task_type = sys.argv[1]
    input_dir = Path(sys.argv[2])
    output_dir = None
    if len(sys.argv) >= 4:
        output_dir = Path(sys.argv[3])

    input_dict = load_input(list(input_dir.glob('*.json')))
    input_sentences = {f: obj['Sentences'] for f, obj in input_dict.items()}

    if task_type == '1a':
        predicted_cue_words = predict_cue_words(input_sentences)
        cue_prediction_dict = predict_cue_links(input_sentences, predicted_cue_words)
    elif task_type == '1b':
        cue_prediction_dict = {}
        for f, obj in input_dict.items():
            cue_prediction_dict[f] = [{'Cue': an['Cue']} for an in obj['Annotations']]

    full_prediction_dict = predict_roles(input_sentences, cue_prediction_dict)

    if sum(len(obj['Annotations']) if 'Annotations' in obj.keys() else 0 for obj in input_dict.values()) > 0:
        gold_annotations = {f: obj['Annotations'] for f, obj in input_dict.items()}
        print('            precision    recall   f1-score')
        for label_ in SPKATT_GEPADE_LABELS:
            pr, rec, f1 = matching_precision_recall_f1(gold_annotations, full_prediction_dict, classes=[label_])
            print(f'{label_:10}     {pr:.4f}    {rec:.4f}     {f1:.4f}')

        pr, rec, f1 = matching_precision_recall_f1(gold_annotations, full_prediction_dict,
                                                   classes=set(SPKATT_GEPADE_LABELS) - {'Cue'})
        print(f'{"Roles":10}     {pr:.4f}    {rec:.4f}     {f1:.4f}')
        pr, rec, f1 = matching_precision_recall_f1(gold_annotations, full_prediction_dict)
        print(f'{"Overall":10}     {pr:.4f}    {rec:.4f}     {f1:.4f}')

    if output_dir is not None:
        print('writing predictions to ', str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        for f, pred in full_prediction_dict.items():
            output_obj = {'Sentences': input_sentences[f], 'Annotations': pred}
            with (output_dir / f).open('w') as output_file:
                json.dump(output_obj, output_file, indent=2)
