
# coding: utf-8

# In[1]:

import numpy as np
import os
import tensorflow as tf


# In[2]:

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# In[3]:

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


# In[4]:

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


# In[5]:

def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


# In[6]:

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:

            tag_name = idx_to_tag[tok]
            tok_chunk_class = tag_name.split('-')[0]
            tok_chunk_type = tag_name.split('-')[-1]

            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


# In[13]:

class NERmodel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.sess   = None
        self.saver  = None
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def build(self):
        #Step1
	#placeholder를 define한다. placeholder는 사용할 데이터들이 들어갈 자리(entries to computational graph)이다.
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")#sequence_length는 sequence labeling을 위한 sequence의 길이를 의미한다.
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")#labels는 입력받은 sequence에 해당되는  label을 의미한다. sequence labeling의 결과가 들어간다.
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")#dropout은 hidden units을 줄이기 위해 사용된다. 
	#reduces complex co-adaptations of neurons, learns robust features in conjunction with many different subsets of other neurons, and avoids overfitting.
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")#lr은 learning rate의 약자이다. train할 때 이용한다.
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")#word id는 word embedding에서 사용 된다. 즉, sequence labeling을 하는데 필요한 요소이다.
        
        #Step2
        with tf.variable_scope("words"):
            
            if self.config.use_pretrained is False:#미리 학습된 embedding이 없는 경우
                print("Randomly initializing word vectors")#random하게 initialize한다.
                _word_embeddings = tf.get_variable(#embedding 정보를 저장하는tensor에 random하게 get
                name = "_word_embeddings",
                dtype = tf.float32,
                shape = [self.config.nwords, self.config.dim_word])
            
            else:#미리 학습된 embedding이 있는 경우, 이를 load한다.
                print("Using pre-trained word vectors : "+sewlf.config.filename_embedding)
                _word_embeddings = tf.Variabel(#embedding 정보를 저장하는 tensor에 미리 학습된 self.config.embeddings를 get
                self.config.embeddings,
                name = "_word_embeddings",
                dtype = tf.float32)
            
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,#embedding은 tensorflow library를 사용하여 생성
                                                    self.word_ids, name="word_embeddings")
        self.word_embeddings = word_embeddings#생성된 embedding은 self.word_embeddings에 저장
        
        #Step3
	#모델을 생성한다. 모델은 Bi-drectional LSTM이다.
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)#foward LSTM cell을 생성
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)#backward LSTM cell을 생성
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(#tensorflow 내장 함수를 통해 LSTM cell에서 결과를 받아온다. 각 output에 저장.
        cell_fw, cell_bw, self.word_embeddings,#tf.nn.bidirectional_dynamic_rnn에 넣어줄 인자는 두 LSTM cell과 word_embeddings, sequence_lengths이다.
        sequence_length = self.sequence_lengths, dtype=tf.float32)
        
        output = tf.concat([output_fw, output_bw], axis=-1)#각 output을 연결
        output = tf.nn.dropout(output, self.dropout)#dropout을 통해 hidden 수를 줄인다. (optional) 
        
        nsteps = tf.shape(output)[1]#output 값의 1-D integer tensor representing the shape의 두번째 값을 nsteps에 저장해 놓는다.
        
        #Step4
	#출력 레이어를 정의한다. 최종 예측 값은 output*W + b이다.
        output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])#output 값을 reshape해준다. 두 LSTM cell 값 취하므로 2*hidden_size_lstm
        W = tf.get_variable("W", dtype=tf.float32,#weight 값을 얻어온다. shape는 2*hidden_size_lstm by self.config.ntags 이다.(pred-예측값 계산을 위한 shape)
                           shape = [2*self.config.hidden_size_lstm, self.config.ntags])
        b = tf.get_variable("b", dtype=tf.float32,#bias 값을 얻어 온다. shape는 slef.config.ntags. 참고로 self.config.ntags는 tag의 개수(len(self.vocab_tags))
                           shape = [self.config.ntags], initializer = tf.zeros_initializer())
        pred = tf.matmul(output, W) + b#위에 정의된 output, weight, bias를 계산하여 최종 예측 값 pred를 계산한다.
        
        self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])#그리고 pred값을 reshape하여 self.logits에 저장한다.
        
        #Step5
	#Loss를 계산한다. 예측값과 정답값을 cross entropy를 이용하여 loss를 구한다.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(#tensorflow library의 내장 함수를 이용한다.
        logits=self.logits, labels = self.labels)#앞서 저장한 pred값(logits)를 파라미터로 갖는다.
        self.loss = tf.reduce_mean(losses)#구한 loss의 reduce mean값을 self.loss에 저장한다.
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)#위를 바탕으로 label의 prediction을 구한다.
        
        #Step6
	#사용할 Optimizer를 정하고 Session을 초기화 및 run한다.
        optimizer = tf.train.AdamOptimizer(self.lr)#Adam Optimizer를 사용한다. Adam은 Stochastic Gradient Descent의 변형 버전 중 하나이다. (미리 세팅된 learning rate 사용)
        self.train_op = optimizer.minimize(self.loss)#loss를 최소화하는 방향으로 학습 진행
        self.sess = tf.Session()#Session을 초기화하고
        self.sess.run(tf.global_variables_initializer())#그 Session을 run한다.
        self.saver = tf.train.Saver()#to save and restore variables to and from checkpoints
        

    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping

        for epoch in range(self.config.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            #score = self.run_epoch(train, dev, epoch)

            #############3
            batch_size = self.config.batch_size
            #nbatches = (len(train) + batch_size - 1)

            # iterate over dataset
            for i, (words, labels) in enumerate(minibatches(train, batch_size)):
                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                           self.config.dropout)

                _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=fd)

                # print(i + 1, [("train loss", train_loss)])

            metrics = self.run_evaluate(dev)
            print("acc : " + str('%.2f' % metrics['acc']) + " - " + "f1 : " + str('%.2f' % metrics['f1']))

            score = metrics["f1"]
            ########## ju_edit
            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                if not os.path.exists(self.config.dir_model):
                    os.makedirs(self.config.dir_model)
                self.saver.save(self.sess, self.config.dir_model)
                best_score = score
                print ("new best score")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print ("- early stopping {} epochs without ""improvement".format(nepoch_no_imprv))
                    break
    
    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        print("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def run_evaluate(self, test):

        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * f1}

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data

        word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred, sequence_lengths

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds


# In[ ]:




# In[ ]:



