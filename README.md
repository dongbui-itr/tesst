# ECG_CAPTIONING



## Prerequisites
```
pip install -r requirements.txt
```

- Download standford corenlp via this link https://stanfordnlp.github.io/CoreNLP/index.html

## Prepare dataset

1. Prepare a csv file contains at least two columns [id, comments], ex: raw_comments.csv
2. Run file processing_data/correct_spelling.py to correct spelling.
   - Replace fields "raw_comments_csv" and "corrected_spelling_comments_csv" in config.json
3. Run file processing_data/gen_data.py to split train, validation and test dataset.
   - Replace fields "train_labels_csv", "val_labels_csv", "test_labels_csv" in config.json
4. If want to augment data, run file processing_data/augmentation.py
   - Replace fields "augmented_file_csv", "file_need_to_augment_csv" in config.json

Note: Full step for "end to end" running. Just steps 1, 2, 3 for testing.

## Training and Testing

- Run file **main.py** to train model. Checkpoints are saved in folder training. Confusion matrix of each epoch is saved in file results/log_val_cf.csv
- Run file **test.py** to test model (using checkpoint name in folder training). Test results are saved in folder results.
