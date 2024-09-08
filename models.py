import tensorflow as tf
from evaluation import compute_wss
from collections import OrderedDict
from operator import itemgetter
from tensorflow.keras import backend as K


class ModelBuilder:
    # better make args dict
    def __init__(self, model, tokenizer, loss_fn='binary_crossentropy', metric='accuracy', epochs=10, batch_size=2):
        self.dropout = 0.2
        self.att_dropout = 0.2
        self.max_length = 512
        self.layer_dropout = 0.2
        self.learning_rate = 5e-5
        self.random_state = 42
        self.transformer_model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.trained_model = None
        self.model_history = None
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               mode='min',
                                                               min_delta=0,
                                                               patience=0,
                                                               restore_best_weights=True)

    def batch_encode(self, texts, batch_size=256):
        """""""""
        A function that encodes a batch of texts and returns the texts'
        corresponding encodings and attention masks that are ready to be fed 
        into a pre-trained transformer model.

        Input:
            - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
            - texts:       List of strings where each string represents a text
            - batch_size:  Integer controlling number of texts in a batch
            - max_length:  Integer controlling max number of words to tokenize in a given text
        Output:
            - input_ids:       sequence of texts encoded as a tf.Tensor object
            - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
        """""""""

        input_ids = []
        attention_mask = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer.batch_encode_plus(batch,
                                                      max_length=self.max_length,
                                                      padding='longest',  # implements dynamic padding
                                                      truncation=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=False
                                                      )
            input_ids.extend(inputs['input_ids'])
            attention_mask.extend(inputs['attention_mask'])

        return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

    def build_model(self):
        """
            Template for building a model off of the BERT or DistilBERT architecture
            for a binary classification task.

            Input:
              - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                              with no added classification head attached.
              - max_length:   integer controlling the maximum number of encoded tokens
                              in a given sequence.

            Output:
              - model:        a compiled tf.keras.Model with added classification layers
                              on top of the base pre-trained model architecture.
            """

        # Define weight initializer with a random seed to ensure reproducibility
        weight_initializer = tf.keras.initializers.GlorotNormal(seed=self.random_state)

        # Define input layers
        input_ids_layer = tf.keras.layers.Input(shape=(self.max_length,),
                                                name='input_ids',
                                                dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(self.max_length,),
                                                      name='input_attention',
                                                      dtype='int32')

        # DistilBERT outputs a tuple where the first element at index 0
        # represents the hidden-state at the output of the model's last layer.
        # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = self.transformer_model([input_ids_layer, input_attention_layer])[0]

        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        # Splicing out the [CLS] tokens gives us 2D data.
        cls_token = last_hidden_state[:, 0, :]

        ##                                                 ##
        ## Define additional dropout and dense layers here ##
        ##                                                 ##

        # Define a single node that makes up the output layer (for binary classification)
        output = tf.keras.layers.Dense(1,
                                       activation='sigmoid',
                                       kernel_initializer=weight_initializer,
                                       kernel_constraint=None,
                                       bias_initializer='zeros'
                                       )(cls_token)

        # Define the model
        model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

        # Compile the model
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self.loss_fn,
                      metrics=self.metric)

        return model

    def train_model(self, x_train_ids, x_train_attention, y_train, x_test_ids, x_test_attention, y_test):
        model = self.build_model()
        NUM_STEPS = len(y_train) // self.batch_size
        self.model_history = model.fit(
            x=[x_train_ids, x_train_attention],
            y=y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=NUM_STEPS,
            validation_data=([x_test_ids, x_test_attention], y_test),
            verbose=2,
            callbacks=[self.early_stopping]
        )
        self.trained_model = model

    def evaluate_model(self, x_test_ids, x_test_attention, y_test):
        y_pred = self.trained_model.predict([x_test_ids, x_test_attention])
        scores = [pred[0] for pred in y_pred]
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
        K.clear_session()
        print("Average WSS@95:", wss_95)
        print("Average WSS@100:", wss_100)
        return wss_95
