import collections
import itertools

import numpy as np

from .common import make_sentence_dataframe, to_idx

CUE_SENTENCE_PADDING = 3
ROLE_SENTENCE_PADDING = 3


def get_word_id_from_input_seq(input_seq):
    previous_word_idx = None
    i = 0
    for word_idx in input_seq.word_ids():
        if word_idx is None:
            yield None
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            yield word_idx
        else:
            yield None
        previous_word_idx = word_idx


def gen_role_sequence(tokenizer, sentence_objects, annotation_objects, add_labels=False, id2label=None, fname=None, return_coords=False):
    if add_labels and id2label is None:
        raise ValueError()

    if id2label is not None:
        num_labels = len(id2label)
        label_idx = [id2label[i] for i in range(num_labels)]

    df = make_sentence_dataframe(sentence_objects, fname)
    for anno in annotation_objects:
        if anno == {}:
            continue

        assert len(anno['Cue']) > 0

        sentid = list(sorted(set(int(x.split(':')[0]) for x in anno['Cue'])))[0]

        segment = df.loc[(fname, slice(sentid - ROLE_SENTENCE_PADDING, sentid + ROLE_SENTENCE_PADDING)), :].copy()
        for k in anno.keys():
            segment[k] = np.array([f'{sentid}:{tokid}' in anno[k] for _, sentid, tokid in segment.index])


        out_words = []
        for cue, x in itertools.groupby(zip(segment.index, segment['Cue']), key=lambda x: x[1]):
            idx, _ = zip(*x)
            if not cue:
                out_words.extend(segment.loc[list(idx), 'token'].values)
            else:
                out_words.append('[LABEL]')
                out_words.extend(segment.loc[list(idx), 'token'].values)
                out_words.append('[CLS]')

        input_seq = tokenizer(out_words, is_split_into_words=True, padding=True)
        if len(input_seq['input_ids']) > tokenizer.max_len_single_sentence:
            print('sentence too long, truncating')
            input_seq = tokenizer(out_words, add_special_tokens=True, is_split_into_words=True, truncation=True)

        if return_coords:
            token_coord_subwords = []
            for word_idx in get_word_id_from_input_seq(input_seq):
                if word_idx is None:  # i.e. subword is either special token or ##-token
                    token_coord_subwords.append(None)
                else:
                    token_coord_subwords.append(segment.index[word_idx])

        if add_labels:
            out_labels_subwords = []
            for word_idx in get_word_id_from_input_seq(input_seq):
                if word_idx is None:  # i.e. subword is either special token or ##-token
                    out_labels_subwords.append([-100] * num_labels)
                else:
                    word_coordinate = segment.index[word_idx]
                    multi_label_arr = segment.loc[word_coordinate, label_idx].values.astype(int)
                    out_labels_subwords.append(multi_label_arr)

            input_seq['labels'] = np.array(out_labels_subwords, dtype=int)
            assert input_seq['labels'].shape == (len(input_seq['input_ids']), len(label_idx))

        if return_coords:
            yield input_seq, token_coord_subwords
        else:
            yield input_seq


def get_cue_sequence(tokenizer, sentence_objects, annotation_objects=None, add_labels=None, fname=None,
                     return_coords=False):
    if add_labels and annotation_objects is None:
        raise ValueError()

    df = make_sentence_dataframe(sentence_objects, fname=fname)
    if add_labels:
        df['is_cue'] = 0
        for obj in annotation_objects:
            for coord_str in obj['Cue']:
                df.loc[to_idx(fname, coord_str), 'is_cue'] = 1

    for sentid in range(max(df.reset_index()['sentid'])):
        segment = df.loc[fname, slice(sentid - CUE_SENTENCE_PADDING, sentid + CUE_SENTENCE_PADDING), slice(None)].copy()
        input_seq = tokenizer(list(segment['token']), add_special_tokens=True, is_split_into_words=True)
        if len(input_seq['input_ids']) > tokenizer.max_len_single_sentence:
            print('sentence too long, truncating')
            input_seq = tokenizer(list(segment['token']), add_special_tokens=True, is_split_into_words=True, truncation=True)

        if return_coords:
            token_coord_subwords = []
            for word_idx in get_word_id_from_input_seq(input_seq):
                if word_idx is None:  # i.e. subword is either special token or ##-token
                    token_coord_subwords.append(None)
                else:
                    token_coord_subwords.append(segment.index[word_idx])

        if add_labels:
            out_labels_subwords = []
            for word_idx in get_word_id_from_input_seq(input_seq):
                if word_idx is None:  # i.e. subword is either special token or ##-token
                    out_labels_subwords.append(-100)
                else:
                    word_coordinate = segment.index[word_idx]
                    is_cue = segment.loc[word_coordinate]['is_cue']
                    if is_cue:
                        out_labels_subwords.append(1)
                    else:
                        out_labels_subwords.append(0)

            input_seq['labels'] = out_labels_subwords

        if return_coords:
            yield input_seq, token_coord_subwords
        else:
            yield input_seq


def get_cue_link_sequence(tokenizer, sentence_objects, annotation_objects=None, positive_cues=None, add_labels=False, fname=None, return_coords=False):
    if annotation_objects is not None and positive_cues is not None:
        raise ValueError()
    if add_labels and annotation_objects is None:
        raise ValueError()

    df = make_sentence_dataframe(sentence_objects, fname=fname)

    if annotation_objects:
        positive_cues = set()
        cue_to_span_id = {}
        for annotation_id, obj in enumerate(annotation_objects):
            for coord_str in obj['Cue']:
                coord = to_idx(fname, coord_str)
                cue_to_span_id[coord] = annotation_id
                positive_cues.add(coord)

    for sentid in range(max(df.reset_index()['sentid'])):
        segment = df.loc[fname, slice(sentid - CUE_SENTENCE_PADDING, sentid + CUE_SENTENCE_PADDING), slice(None)].copy()
        cues_in_sent = [(f_, sentid_, tokid_) for f_, sentid_, tokid_ in positive_cues if (f_, sentid_) == (fname, sentid)]
        for coord_1, coord_2 in itertools.combinations(cues_in_sent, 2):
            out_words = []
            for idx in segment.index:
                if idx not in [coord_1, coord_2]:
                    out_words.extend(segment.loc[list(idx), 'token'].values)
                else:
                    out_words.append('[LABEL]')
                    out_words.extend(segment.loc[list(idx), 'token'].values)
                    out_words.append('[CLS]')

            input_seq = tokenizer(out_words, is_split_into_words=True, padding=True)
            if len(input_seq['input_ids']) > tokenizer.max_len_single_sentence:
                print('sentence too long, truncating')
                input_seq = tokenizer(out_words, add_special_tokens=True, is_split_into_words=True, truncation=True)

            if add_labels:
                if cue_to_span_id[coord_1] == cue_to_span_id[coord_2]:
                    out_label = 1
                else:
                    out_label = 0
                input_seq['label'] = out_label

            if return_coords:
                return input_seq, (coord_1, coord_2)
            else:
                return input_seq




