"""
reference: https://github.com/HUIYINXUE/hashformer/blob/master/src/utils/free_tokenizer.py
"""
import re
import json


special_tokens = {
            '<s>': 0,
            '<pad>': 1,
            '</s>': 2,
            '<unk>': 3,
            '<mask>': 4,           
            '<num>': 5,
            'a': 6,
            'i': 7,
            '.': 8,
            '!': 9,
            '?': 10,
            ',': 11,
            ';': 12
        }

def build_vocab(char_num):
    chars = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    vocab = special_tokens
    token_id = len(vocab)

    if char_num == 1:
        # build 1 char vocab
        for c in chars:
            if c not in vocab:
                vocab[c] = token_id
                token_id += 1
        
        return vocab

    if char_num > 1:
        # build 2 chars vocab
        for c1 in chars:
            for c2 in chars:
                if c1 + c2 not in vocab:
                    vocab[c1 + c2] = token_id
                    token_id += 1

    if char_num > 2:
        # build 3 chars vocab
        for c1 in chars:
            for c2 in chars:
                for c3 in chars:
                    if c1 + c2 + c3 not in vocab:
                        vocab[c1 + c2 + c3] = token_id
                        token_id += 1

    return vocab

class Tokenizer_f():
    """
    Tokenizer for words first character
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=1)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the first char if not a special token
            if token not in self.vocab and len(token) > 1:
                token = token[0]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]



class Tokenizer_m():
    """
    Tokenizer for words mid character
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=1)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the mid char if not a special token
            if token not in self.vocab and len(token) > 1:
                token = token[len(token)//2]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_l():
    """
    Tokenizer for words last character
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=1)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the last char if not a special token
            if token not in self.vocab and len(token) > 1:
                token = token[-1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_ff():
    """
    Tokenizer for words first two characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=2)


    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id.
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the first 2 chars if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[0] + token[1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_ll():
    """
    Tokenizer for words last two characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=2)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the last 2 chars if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[-2] + token[-1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_fl():
    """
    Tokenizer for words first and last characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=2)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the first and last char if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[0] + token[-1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])  
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]



class Tokenizer_fff():
    """
    Tokenizer for words first three characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=3)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the first 3 characters if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[0] + token[1] + token[2]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])   
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_lll():
    """
    Tokenizer for words last three characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=3)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the last 3 characters if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[-3] + token[-2] + token[-1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])  
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_fml():
    """
    Tokenizer for words first, middle and last characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.vocab = build_vocab(char_num=3)

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        elif isinstance(text, tuple):
            # process a tuple
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())
    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id. 
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # get the first, mid and last characters if not a special token
            if token not in self.vocab and len(token) > 2:
                token = token[0] + token[len(token)//2] + token[-1]

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_full_token():
    """
    Tokenizer for full word tokens
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.vocab = {}
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.build_vocab()

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def build_vocab(self):
        f = open('vocab_full_tokens.json')
        self.vocab = json.load(f)

    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id.
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_full_token_vowels():
    """
    Tokenizer for words vowel characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.vocab = {}
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.build_vocab()

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def build_vocab(self):
        f = open('vocab_full_tokens_no_consonants.json')
        self.vocab = json.load(f)

    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id.
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # remove vowels
            if token not in self.vocab and len(token) > 1:
                token = re.sub(r'[^aeiouAEIOU]', '', token)

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]


class Tokenizer_full_token_consonants():
    """
    Tokenizer for words consonant characters
    """
    def __init__(self, **kwargs):
        self.reg = r"\w+|[^\w\s]+"
        self.vocab = {}
        self.mask_token="<mask>"
        self.pad_token="<pad>"
        self.pad_token_id = 1
        self.build_vocab()

    def __call__(self, text, max_length=512, **kwargs):
        if isinstance(text, list):
            # process a batch of docs
            candis = list(map(lambda x: self._proc_each(x, max_length), text))
            # using the first element in the list to figure out the keys.
            first = candis[0]
            batch = {}
            for k, v in first.items():
                batch[k] = [f[k] for f in candis]
            return batch
        elif isinstance(text, str):
            # process only one instance
            return self._proc_each(text, max_length)
        else:
            print("The input has to be a string or list!")
            exit(1)

    def _proc_each(self, text, max_length=512):
        if isinstance(text, str):
            # processing single
            sent = self.tokenize(text)[: max_length-2]
            sent_len = len(sent)
            inputs = ['<s>'] + sent + ['</s>'] + ['<pad>'] * (max_length - 2 - sent_len)
            attention_mask = [1] * (sent_len + 2) + [0] * (max_length - 2 - sent_len)
        elif isinstance(text, tuple):
            # processing pair
            sent_1 = self.tokenize(text[0])
            if text[1]:
                sent_2 = self.tokenize(text[1])
            else:
                sent_2 = []    
            len_1 = len(sent_1)
            len_2 = len(sent_2)
            text_len = len_1 + len_2
            thres = max_length // 2 - 2
            if text_len > max_length - 4:
                # tokens occupy the whole sequence
                attention_mask = [1] * max_length
                if len_1 <= len_2:
                    if len_1 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2[:max_length-4-len_1] + ['</s>']
                else:
                    if len_2 >= thres:
                        # both truncate to thres
                        inputs = ['<s>'] + sent_1[:thres] + ['</s>'] * 2 + sent_2[:thres] + ['</s>']
                    else:
                        # short sentence fully reserved
                        inputs = ['<s>'] + sent_1[:max_length - 4 - len_2] + ['</s>'] * 2 + sent_2 + ['</s>']
            else:
                inputs = ['<s>'] + sent_1 + ['</s>'] * 2 + sent_2 + ['</s>'] + ['<pad>'] * (max_length-4-text_len)
                attention_mask = [1] * (text_len + 4) + [0] * (max_length - 4 - text_len)
        else:
            print("Each instance has to be a string or a tuple of string pair!")
            exit(1)
        input_ids = self.convert_tokens_to_ids(inputs)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def tokenize(self, text):
        return re.findall(self.reg, text.lower())

    def build_vocab(self):
        f = open('vocab_full_tokens_no_vowels.json')
        self.vocab = json.load(f)

    
    def get_vocab(self):
        return self.vocab

    def save_vocab(self):
        with open('vocab.json', 'w') as f:
            json.dump(self.vocab, f)

    def convert_tokens_to_ids(self, token_list):
        """
        Function to map each token to its corresponding id.
        """
        token_ids = []
        for token in token_list:
            if token.isdigit():
                token = '<num>'

            # remove vowels
            if token not in self.vocab and len(token) > 1:
                token = re.sub(r'[aeiouAEIOU]', '', token)

            if token not in self.vocab:
                token_ids.append(self.vocab['<unk>'])
            else:
                token_ids.append(self.vocab[token])
            
        
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x < 6 else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]