from __future__ import print_function

import torch
import numpy as np
import os
import math
from typing import List
from torch import optim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from embedded_topic_model.core.nets import Etm, ProdEtm, DropProdEtm
from embedded_topic_model.utils import data, embedding, metrics


class TopicModel:
    """
    Creates an embedded topic model instance. The model hyperparameters are:

        vocabulary (list of str): training dataset vocabulary
        model (embedded_topic_model.core.model.BaseModel): model to train
        embeddings (str or KeyedVectors): KeyedVectors instance containing word-vector mapping for embeddings, or its path
        use_c_format_w2vec (bool): wheter input embeddings use word2vec C format. Both BIN and TXT formats are supported
        model_path (str): path to save trained model. If None, the model won't be automatically saved
        batch_size (int): input batch size for training
        lr (float): learning rate
        lr_factor (float): divide learning rate by this...
        epochs (int): number of epochs to train. 150 for 20ng 100 for others
        optimizer_type (str): choice of optimizer
        seed (int): random seed (default: 1)
        enc_drop (float): dropout rate on encoder
        clip (float): gradient clipping
        nonmono (int): number of bad hits allowed
        wdecay (float): some l2 regularization
        anneal_lr (bool): whether to anneal the learning rate or not
        bow_norm (bool): normalize the bows or not
        num_words (int): number of words for topic viz
        log_interval (int): when to log training
        visualize_every (int): when to visualize results
        eval_batch_size (int): input batch size for evaluation
        eval_perplexity (bool): whether to compute perplexity on document completion task
        debug_mode (bool): wheter or not should log model operations
    """

    def __init__(
        self,
        vocabulary: list,
        model_name: str,
        num_topics: int = 50,
        t_hidden_size: int = 800,
        rho_size: int = 100,
        theta_act: str = 'relu',
        enc_drop=0.2,
        topic_dropout=0.5,
        embeddings="embedded_topic_model/pretrained_embeddings/glove.6B.100d.txt",
        model_path=None,
        batch_size=32,
        train_embeddings=True,
        lr=0.005,
        lr_factor=4.0,
        epochs=100,
        optimizer_type='adam',
        seed=2019,
        clip=0.0,
        nonmono=10,
        wdecay=1.2e-6,
        anneal_lr=False,
        bow_norm=True,
        num_words=10,
        log_interval=2,
        visualize_every=10,
        eval_batch_size=1000,
        eval_perplexity=False,
        debug_mode=False,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.rho_size = rho_size
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.num_topics = num_topics
        self.t_hidden_size = t_hidden_size
        self.theta_act = theta_act
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr_factor = lr_factor
        self.epochs = epochs
        self.seed = seed
        self.enc_drop = enc_drop
        self.topic_dropout = topic_dropout
        self.clip = clip
        self.nonmono = nonmono
        self.anneal_lr = anneal_lr
        self.bow_norm = bow_norm
        self.num_words = num_words
        self.log_interval = log_interval
        self.visualize_every = visualize_every
        self.eval_batch_size = eval_batch_size
        self.eval_perplexity = eval_perplexity
        self.debug_mode = debug_mode

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.train_embeddings = train_embeddings
        self.embeddings = None if train_embeddings else self._initialize_embeddings(embeddings)

        self.model = self._get_model(model_name)
        self.optimizer = self._get_optimizer(optimizer_type, lr, wdecay)

    def _get_model(self, model_name: str):
        if self.debug_mode: print(f"Init a {model_name} instance")
        if model_name == 'etm':
            model = Etm(
                self.vocab_size, self.num_topics, self.t_hidden_size, self.rho_size, self.theta_act,
                self.train_embeddings, self.embeddings, self.enc_drop, self.debug_mode
            )
        if model_name == 'prodetm':
            model = ProdEtm(
                self.vocab_size, self.num_topics, self.t_hidden_size, self.rho_size, self.theta_act,
                self.train_embeddings, self.embeddings, self.enc_drop, self.debug_mode
            )
        if model_name == 'dropprodetm':
            model = DropProdEtm(
                self.vocab_size, self.num_topics, self.t_hidden_size, self.rho_size, self.theta_act,
                self.train_embeddings, self.embeddings, self.enc_drop, self.topic_dropout, self.debug_mode
            )
        return model.to(self.device)

    def __str__(self):
        return f'{self.model}'

    def _get_extension(self, path):
        assert isinstance(path, str), 'path extension is not str'
        filename = path.split(os.path.sep)[-1]
        return filename.split('.')[-1]

    def _initialize_embeddings(
        self, 
        embeddings
    ):
        vectors = embeddings if isinstance(embeddings, KeyedVectors) else {}
        
        try:
            vectors = KeyedVectors.load_word2vec_format(embeddings)
        except:
            if self.debug_mode: print("Converting GloVe format to W2V format...")
            w2v_output_path = ".".join(embeddings.split(".")[:-1]) + ".w2v"
            _ = glove2word2vec(embeddings, w2v_output_path)
            vectors = KeyedVectors.load_word2vec_format(w2v_output_path)
        if self.debug_mode: print("Load gensim key vector file successfully!")

        model_embeddings = np.zeros((self.vocab_size, self.rho_size))

        for i, word in enumerate(self.vocabulary):
            try:
                model_embeddings[i] = vectors[word]
            except KeyError:
                model_embeddings[i] = np.random.normal(
                    scale=0.6, size=(self.rho_size, ))
        return torch.from_numpy(model_embeddings).to(self.device)

    def _get_optimizer(self, optimizer_type, learning_rate, wdecay):
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'adadelta':
            return optim.Adadelta(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'asgd':
            return optim.ASGD(
                self.model.parameters(),
                lr=learning_rate,
                t0=0,
                lambd=0.,
                weight_decay=wdecay)
        else:
            if self.debug_mode:
                print('Defaulting to vanilla SGD')
            return optim.SGD(self.model.parameters(), lr=learning_rate)

    def _set_training_data(self, train_data):
        self.train_tokens = train_data['tokens']
        self.train_counts = train_data['counts']
        self.num_docs_train = len(self.train_tokens)

    def _set_test_data(self, test_data):
        self.test_tokens = test_data['test']['tokens']
        self.test_counts = test_data['test']['counts']
        self.num_docs_test = len(self.test_tokens)

    def _train(self, epoch):
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.num_docs_train)
        indices = torch.split(indices, self.batch_size)
        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()

            data_batch = data.get_batch(
                self.train_tokens,
                self.train_counts,
                ind,
                self.vocab_size,
                self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self.model(
                data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()

            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        if self.debug_mode:
            print('Epoch {:<3} \t KL Loss: {:<10.2f} Rec Loss: {:<10.2f} \t NELBO: {:<10.2f}'.format(
                epoch, cur_kl_theta, cur_loss, cur_real_loss))

    def _perplexity(self) -> float:
        """Computes perplexity on document completion for a given testing data.

        The document completion task is described on the original ETM's article: https://arxiv.org/pdf/1907.04907.pdf

        Parameters:
        ===
            test_data (dict): BOW testing dataset, split in tokens and counts and used for perplexity

        Returns:
        ===
            float: perplexity score on document completion task
        """
        self.model.eval()
        with torch.no_grad():
            acc_loss = 0
            cnt = 0
            indices = torch.randperm(self.num_docs_test)
            indices = torch.split(indices, self.eval_batch_size)
            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(
                    self.train_tokens,
                    self.train_counts,
                    ind,
                    self.vocab_size,
                    self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                if self.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                recon_loss, _ = self.model(
                    data_batch, normalized_data_batch)

                acc_loss += torch.sum(recon_loss).item()
                cnt += 1

                if idx % self.log_interval == 0 and idx > 0:
                    cur_loss = round(acc_loss / cnt, 2)

            cur_loss = round(acc_loss / cnt, 2)

            if self.debug_mode:
                print('Document Completion Task Perplexity: {:<10.2f}'.format(cur_loss))

            return cur_loss

    def get_topics(self, top_n_words=10) -> List[str]:
        """
        Gets topics. By default, returns the 10 most relevant terms for each topic.

        Parameters:
        ===
            top_n_words (int): number of top words per topic to return

        Returns:
        ===
            list of str: topic list
        """

        with torch.no_grad():
            topics = []
            betas = self.model.get_beta()

            for k in range(self.model.num_topics):
                beta = betas[k]
                top_words = list(beta.cpu().numpy().argsort()
                                 [-top_n_words:][::-1])
                topic_words = [self.vocabulary[a] for a in top_words]
                topics.append(topic_words)

            return topics

    def get_most_similar_words(
        self, 
        queries=["computer", "sports", "religion", "man", "love", "politics", "health", "people"], n_most_similar=5
    ) -> dict:
        """
        Gets the nearest neighborhoring words for a list of tokens. By default, returns the 20 most similar words for each token in 'queries' array.

        Parameters:
        ===
            queries (list of str): words to find similar ones
            n_most_similar (int): number of most similar words to get for each word given in the input. By default is 20

        Returns:
        ===
            dict of (str, list of str): dictionary containing the mapping between query words given and their respective similar words
        """

        self.model.eval()

        # visualize word embeddings by using V to get nearest neighbors
        with torch.no_grad():
            try:
                self.embeddings = self.model.rho.weight  # Vocab_size x E
            except BaseException:
                self.embeddings = self.model.rho         # Vocab_size x E

            neighbors = {}
            for word in queries:
                neighbors[word] = metrics.nearest_neighbors(
                    word, self.embeddings, self.vocabulary, n_most_similar)
                if self.debug_mode: print(f"Query {word}: {neighbors}")

            return neighbors

    def fit(self, train_data, test_data=None):
        """
        Trains the model with the given training data.

        Optionally receives testing data for perplexity calculation. The testing data is
        only used if the 'eval_perplexity' model parameter is True.

        Parameters:
        ===
            train_data (dict): BOW training dataset, split in tokens and counts
            test_data (dict): optional. BOW testing dataset, split in tokens and counts. Used for perplexity calculation, if activated

        Returns:
        ===
            self (ETM): the instance itself
        """
        self._set_training_data(train_data)
        if test_data is not None: self._set_test_data(test_data)

        best_val_ppl = 1e9
        all_val_ppls = []

        if self.debug_mode:
            print(f'Topics before training: {self.get_topics()}')

        for epoch in range(1, self.epochs + 1):
            self._train(epoch)

            if self.eval_perplexity:
                val_ppl = self._perplexity(
                    test_data)
                if val_ppl < best_val_ppl:
                    if self.model_path is not None:
                        self._save_model(self.model_path)
                    best_val_ppl = val_ppl
                else:
                    # check whether to anneal lr
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.anneal_lr and (len(all_val_ppls) > self.nonmono and val_ppl > min(
                            all_val_ppls[:-self.nonmono]) and lr > 1e-5):
                        self.optimizer.param_groups[0]['lr'] /= self.lr_factor

                all_val_ppls.append(val_ppl)

            if self.debug_mode and (epoch % self.visualize_every == 0):
                print(f'Topics: {self.get_topics()}')

        if self.model_path is not None:
            self._save_model(self.model_path)

        if self.eval_perplexity and self.model_path is not None:
            self._load_model(self.model_path)
            val_ppl = self._perplexity(train_data)

        return self

    def get_topic_word_matrix(self) -> List[List[str]]:
        """
        Obtains the topic-word matrix learned for the model.

        The topic-word matrix lists all words for each discovered topic.
        As such, this method will return a matrix representing the words.

        Returns:
        ===
            list of list of str: topic-word matrix.
            Example:
                [['world', 'planet', 'stars', 'moon', 'astrophysics'], ...]
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta()

            topics = []

            for i in range(self.model.num_topics):
                words = list(beta[i].cpu().numpy())
                topic_words = [self.vocabulary[a] for a, _ in enumerate(words)]
                topics.append(topic_words)

            return topics

    def get_topic_word_dist(self) -> torch.Tensor:
        """
        Obtains the topic-word distribution matrix.

        The topic-word distribution matrix lists the probabilities for each word on each topic.

        This is a normalized distribution matrix, and as such, each row sums to one.

        Returns:
        ===
            torch.Tensor: topic-word distribution matrix, with KxV dimension, where
            K is the number of topics and V is the vocabulary size
            Example:
                tensor([[3.2238e-04, 3.7851e-03, 3.2811e-04, ..., 8.4206e-05, 7.9504e-05,
                4.0738e-04],
                [3.6089e-05, 3.0677e-03, 1.3650e-04, ..., 4.5665e-05, 1.3241e-04,
                5.8661e-05]])
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            return self.model.get_beta()

    def get_document_topic_dist(self) -> torch.Tensor:
        """
        Obtains the document-topic distribution matrix.

        The document-topic distribution matrix lists the probabilities for each topic on each document.

        This is a normalized distribution matrix, and as such, each row sums to one.

        Returns:
        ===
            torch.Tensor: topic-word distribution matrix, with DxK dimension, where
            D is the number of documents in the corpus and K is the number of topics
            Example:
                tensor([[0.1840, 0.0489, 0.1020, 0.0726, 0.1952, 0.1042, 0.1275, 0.1657],
                [0.1417, 0.0918, 0.2263, 0.0840, 0.0900, 0.1635, 0.1209, 0.0817]])
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            indices = torch.tensor(range(self.num_docs_train))
            indices = torch.split(indices, self.batch_size)

            thetas = []

            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(
                    self.train_tokens,
                    self.train_counts,
                    ind,
                    self.vocab_size,
                    self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums if self.bow_norm else data_batch
                theta, _ = self.model.get_theta(normalized_data_batch)

                thetas.append(theta)

            return torch.cat(tuple(thetas), 0)

    def get_topic_coherence(self, top_n=10) -> float:
        """
        Calculates NPMI topic coherence for the model.

        By default, considers the 10 most relevant terms for each topic in coherence computation.

        Parameters:
        ===
            top_n (int): number of words per topic to consider in coherence computation

        Returns:
        ===
            float: the model's topic coherence
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta().data.cpu().numpy()
            return metrics.get_topic_coherence(
                beta, self.train_tokens, self.vocabulary, top_n)

    def get_topic_diversity(self, top_n=25) -> float:
        """
        Calculates topic diversity for the model.

        By default, considers the 25 most relevant terms for each topic in diversity computation.

        Parameters:
        ===
            top_n (int): number of words per topic to consider in diversity computation

        Returns:
        ===
            float: the model's topic diversity
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta().data.cpu().numpy()
            return metrics.get_topic_diversity(beta, top_n)

    def eval(self, metrics=['topic', 'tc', 'td', 'ppl', 'embedding']):
        print(f"Model type      : {self.model_name}")
        print(f"Vocab size      : {self.vocab_size}")
        print(f"Num topics      : {self.num_topics}")
        print(f"Train embedding : {'true' if self.train_embeddings else 'false'}")
        if 'topic' in metrics:      print(f"Topic           : {self.get_topics()}")
        if 'tc' in metrics:         print(f"Topic coherence : {self.get_topic_coherence():<8.4f}")
        if 'td' in metrics:         print(f"Topic diversity : {self.get_topic_diversity()*100:<4.2f}")
        if 'topic' in metrics:      print(f"Topic           : {self.get_topics()}")
        if 'embedding' in metrics:  print(f"Word embedding  : {self.get_most_similar_words()}")

    def save_model(self, model_path):
        assert self.model is not None, \
            'no model to save'

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as file:
            torch.save(self.model, file)

    def load_model(self, model_path):
        assert os.path.exists(model_path), \
            "model path doesn't exists"

        with open(model_path, 'rb') as file:
            self.model = torch.load(file)
            self.model = self.model.to(self.device)