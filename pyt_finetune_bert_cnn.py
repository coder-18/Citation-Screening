from collections import OrderedDict
from operator import itemgetter
import torch.nn as nn
from transformers import BertModel, DistilBertModel, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from augmentation import get_augmentation_text
from evaluation import compute_wss
from preprocessing import load_data, split_train_test_data, load_pico_file
from transformers import DistilBertTokenizerFast, DistilBertConfig, TFDistilBertModel, AutoModel, AutoTokenizer, \
    TFBertModel, BertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, random_split
import logging
import sys
import yaml
import gc

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    augment = cfg['Augmentation']['augment']
    language_translated_aug = cfg['Augmentation']['language_translated_aug']
    augtext_augmentation = cfg['Augmentation']['augtext_augmentation']
    language_translated_aug_dir = cfg['Augmentation']['language_translated_aug_dir']
    model_text = cfg['Model']['model_text']
    pico_dir = cfg['Model']['pico_dir']
    epochs = cfg['Model']['epochs']
    batch_size = cfg['Model']['batch_size']


# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    logging.info(f'There are  {torch.cuda.device_count()} GPU(s) available.')

    logging.info(f'Running on the GPU:- {torch.cuda.get_device_name(0)}')
else:
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, model_name, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2
        # Freeze the BERT model

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if model_name == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
            self.bert = DistilBertModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
            self.bert = BertModel.from_pretrained(model_name)
        # Instantiate BERT model
        self.hidden_size = self.bert.config.hidden_size
        self.conv = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding='valid', stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=64 - 5 + 1)
        self.dropout = nn.Dropout(0.3)
        self.clf1 = nn.Linear(2048, 2)

    def forward(self, inputs, mask):
        cls_hs = self.bert(input_ids=inputs, attention_mask=mask, return_dict=False)
        x = cls_hs[0]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.clf1(x)
        return x





class ModelBuilderPt(nn.Module):
    # better make args dict
    def __init__(self, model_name, max_len=512, labels_size=2, epochs=4, batch_size=2, freeze_bert=False):

        self.max_len = max_len
        self.batch_size = batch_size
        if model_name == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.epochs = epochs

    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every text...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def initialize_model(self, model_name, train_dataloader):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        bert_classifier = BertClassifier(model_name, freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        bert_classifier.to(device)

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )
        bert_classifier.optimizer = optimizer
        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)
        bert_classifier.scheduler = scheduler
        return bert_classifier, optimizer, scheduler

    def train(self, model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
        """Train the BertClassifier model.
        """

        # Start training loop
        logging.info("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            logging.info(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            logging.info("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = model.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                model.optimizer.step()
                model.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    logging.info(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            logging.info("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                logging.info(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                logging.info("-" * 70)
            logging.info("\n")

        logging.info("Training complete!")
        return model

    def evaluate(self, model, val_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = model.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def bert_predict(self, model, test_dataloader, y_test):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        scores = [pred[1] for pred in probs]
        test_indexes_with_distances = {}
        for index, prediction in enumerate(scores):
            test_indexes_with_distances[index] = scores[index]

        test_indexes_with_distances = OrderedDict(
            sorted(
                test_indexes_with_distances.items(), key=itemgetter(1), reverse=True
            )
        )
        wss_95, wss_100, precision_95 = compute_wss(
            indexes_with_predicted_distances=test_indexes_with_distances,
            y_test=y_test,
        )
        # logging.info(f"y_test values {y_test}")
        # logging.info(f"predictions values {probs}")
        logging.info(f"Average WSS@95:  {wss_95}")
        logging.info(f"Average WSS@100:  {wss_100}")

        return probs, wss_95


def add_augmentation(x_train, input_file_name):
    file_name = input_file_name.split('/')[-1]
    file_name = language_translated_aug_dir + 'augmented_' + file_name
    csv_file = file_name
    augtexts = []
    translations = []
    if language_translated_aug:
        translations = get_augmentation_text(x_train, csv_file)
    if augtext_augmentation:
        augtexts = []
    augmentations = translations + augtexts
    return augmentations


def train_and_evaluate_model(model_name='distilbert-base-uncased', input_file_name='data/ADHD.tsv', seed=42):
    logging.info(f'setting up seed -> {seed}')
    set_seed(seed_value=seed)
    logging.info(f'Splitting and preprocessing  the data')
    if model_text == 'Pico_sentences':
        # update_filename here
        _, y = load_data(input_file_name)
        logging.info(f'augmenting pico sentences')
        input_file_name_pico = input_file_name.split('/')[-1]
        input_file_name_pico = pico_dir + 'pico_' + input_file_name_pico
        X = load_pico_file(input_file_name_pico)
    else:
        X, y = load_data(input_file_name)

    x_train, x_test, y_train, y_test = split_train_test_data(X, y, test_size=0.5, seed=seed)
    if augment and language_translated_aug and model_text != 'Pico_sentences':  # cant happen together
        logging.info(f'augmenting language translations')
        augmentations = add_augmentation(x_train, input_file_name)
        x_train = np.array(x_train.tolist() + augmentations)
        y_train = np.array(y_train.tolist() + [1] * (len(augmentations)))

    model_clf = ModelBuilderPt(model_name=model_name, epochs=epochs, batch_size=batch_size)

    x_train_ids, x_train_attention = model_clf.preprocessing_for_bert(x_train.tolist())
    x_test_ids, x_test_attention = model_clf.preprocessing_for_bert(x_test.tolist())

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_test)

    # Create the DataLoader for our training set
    train_data = TensorDataset(x_train_ids, x_train_attention, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=model_clf.batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(x_test_ids, x_test_attention, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=model_clf.batch_size)

    bert_classifier, optimizer, scheduler = model_clf.initialize_model(model_name, train_dataloader)
    bert_classifier.loss_fn = nn.CrossEntropyLoss()
    bert_classifier.optimizer = optimizer
    bert_classifier.scheduler = scheduler
    model = model_clf.train(bert_classifier, train_dataloader, val_dataloader, epochs=epochs, evaluation=True)
    preds, wss_95 = model_clf.bert_predict(model, val_dataloader, y_test)

    return wss_95

# if __name__ == '__main__':
#     set_seed(seed_value=42)
#     input_data_file = 'data/ADHD.tsv'
#     model_name = 'distilbert-base-uncased'
#     X, y = load_data(input_data_file)
#     x_train, x_test, y_train, y_test = split_train_test_data(X, y, test_size=0.5, seed=seed)
#     model_clf = ModelBuilderPt(model_name=model_name)
#
#     x_train_ids, x_train_attention = model_clf.preprocessing_for_bert(x_train.tolist())
#     x_test_ids, x_test_attention = model_clf.preprocessing_for_bert(x_test.tolist())
#
#     # Convert other data types to torch.Tensor
#     train_labels = torch.tensor(y_train)
#     val_labels = torch.tensor(y_test)
#
#     # Create the DataLoader for our training set
#     train_data = TensorDataset(x_train_ids, x_train_attention, train_labels)
#     train_sampler = RandomSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=model_clf.batch_size)
#
#     # Create the DataLoader for our validation set
#     val_data = TensorDataset(x_test_ids, x_test_attention, val_labels)
#     val_sampler = SequentialSampler(val_data)
#     val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=model_clf.batch_size)
#
#     bert_classifier, optimizer, scheduler = model_clf.initialize_model(model_name, train_dataloader)
#     model = model_clf.train(bert_classifier, train_dataloader, val_dataloader, epochs=4, evaluation=True)
#
#     preds, wss_95 = model_clf.bert_predict(model, val_dataloader, y_test)
#     print(preds)
#     print(val_labels)
#     print(wss_95)
# files = glob.glob('/content/Project_dir/data/*.tsv')
