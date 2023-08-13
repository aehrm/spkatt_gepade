# Speaker Attribution in German Parliamentary Debates through BERT models

This repository holds the code for the submission “Politics, BERTed: Automatic
Attribution of Speech Events in German Parliamentary Debates” submitted to the
[KONVENS 2023 Shared Task on Speaker Attribution](https://github.com/umanlp/SpkAtt-2023), Task 1.

The task is divided into two subtasks:
* Task 1a is the full task, predicting both cue spans and associated role spans
* Task 1b is the role prediction task only, where gold cue spans are already given.

The goal of the shared task is the automatic identification of speech events in
political debates  and attributing them to their respective speakers,
essentially identifying who says what to whom in the parliamentary debates.

## Models

| Used Base Model                                                   | SpkAtt-F1 (test set) | Match-F1 (dev set) | Download                   |
|-------------------------------------------------------------------|----------------------|--------------------|----------------------------|
| [aehrm/gepabert](https://huggingface.co/aehrm/gepabert)           | 82.8                 | 84.8               | [Link][gepabert-release]   |
| [deepset/gbert-large](https://huggingface.co/deepset/gbert-large) | (not evaluated)      | 84.4               | [Link][gbertlarge-release] |
| [deepset/gbert-base](https://huggingface.co/deepset/gbert-base)   | (not evaluated)      | 81.2               | [Link][gbertbase-release]  |


## Setup

The project uses poetry for dependency management. You can just run:
`poetry install` to install all dependencies.

You may open a shell with `poetry shell` with all required python packages and interpreter.
Alternatively, you can run scripts with the project-dependent python interpreter with `poetry run python <script.py>`.

## Usage

### Inference

Before inference, you either need to [download the published models][gepabert-release] and
place them into the `models/` folder, or train the models yourself (see below).

After the `models/` folder has been populated, you can run the full inference (1a) like this:
```sh
# adjust if needed
#export MODEL_OUTPUT_DIR=./models

poetry run python ./predict.sh 1a input_dir [output_dir]
```

The `input_dir` should hold tokenized speeches as JSON file, like in the [GePaDe test dataset](https://github.com/umanlp/SpkAtt-2023/tree/master/data/eval/task1) (the one provided for
the shared task).

Alternatively, you can run the subtask 1b (role prediction from gold cues) like the following, e.g., on
the [GePaDe dev dataset](https://github.com/umanlp/SpkAtt-2023/tree/master/data/dev/task1). Make sure the input_dir JSON files contain annotation objects with cue spans.
```sh
poetry run python ./predict.sh 1b path/to/spkatt_data/dev/task1 [output_dir]
```

### Training

After downloading the [full GePaDe dataset](https://github.com/umanlp/SpkAtt-2023/tree/master/data) in the folder `data`, you can run the training like this:
```sh
# adjust if needed
#export BASE_MODEL_NAME=aehrm/gepabert
#export MODEL_OUTPUT_DIR=./models
#export TRAIN_FILES='./data/train/task1'
#export DEV_FILES='./data/dev/task1'

poetry run python ./predict.sh 1a input_dir [output_dir]
```

[gepabert-release]: https://github.com/aehrm/spkatt_gepade/releases/download/konvens/gepabert_models.tar
[gbertlargt-release]: https://github.com/aehrm/spkatt_gepade/releases/download/konvens/gbertbase_models.tar
[gbertbast-release]: https://github.com/aehrm/spkatt_gepade/releases/download/konvens/gbertlarge_models.tar
