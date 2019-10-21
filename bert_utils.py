from pytorch_pretrained_bert import BertTokenizer
import csv
from pytorch_transformers.optimization import AdamW
from torch.utils.data import Dataset, DataLoader

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

class InputExample(object):

    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label


class DataGenerator(object):

    def __init__(self, labels, data_dir, input_file):
        self.labels = labels
        self.data_dir = data_dir
        self.input_file = input_file

    def _read_excel(self, input_file, data_dir):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, set_type):
        return self._create_examples(self._read_excel(self.input_file, self.data_dir), "train")


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_example_to_feature(example_row):
    example, label_map, max_seq_length, tokenizer = example_row

    tokens = tokenizer.tokenize(example.text_a)
    tokens = tokens[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


def make_train_step(model, loss_fct, optimizer):
    def train_step(data_loader):
        _ = model.train()
        for step, batch in enumerate(tqdm_notebook(data_loader, desc="Iteration")):
            input_ids, input_mask = batch['all_input_ids'], batch['all_input_mask']
            segment_ids, label_ids = batch['all_segment_ids'], batch['all_label_ids']
            cont_data = batch['cont_cols']
            cat_data = {key: batch[key] for key in batch.keys() if key not in ['all_input_ids', 'all_label_ids'
                                                                                                'all_input_mask',
                                                                               'all_segment_ids', 'cont_cols']}
            logits = model.forward(cont_data, cat_data, input_ids, segment_ids, input_mask, labels=None)

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return step, loss.item()

    return train_step


class Load_Data(Dataset):

    def __init__(self, data, cat_cols, cont_cols, features):

        self.n = data.shape[0]
        self.data_dict = {}

        self.data_dict['all_input_ids'] = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.data_dict['all_input_mask'] = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.data_dict['all_segment_ids'] = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.data_dict['all_label_ids'] = torch.tensor([f.label_id for f in features], dtype=torch.long)

        for col in cat_cols:
            self.data_dict[col] = torch.tensor(data[col], dtype=torch.long)

        self.data_dict['cont_cols'] = torch.tensor(data[cont_cols].values, dtype=torch.float)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        output = {}
        for key in self.data_dict.keys():
            output[key] = self.data_dict[key][idx]

        output['cont_cols'] = self.data_dict['cont_cols'][idx]

        return output