import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

class NLIPreprocess:
    def __init__(self, path, tokenizer_path='../pretrained/sarvam-1/', max_length=150, num_rows=None):
        self.ds = load_dataset("csv", data_files=path)['train']
        if num_rows is not None:
            self.ds = self.ds.select(range(min(num_rows, len(self.ds))))

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_eos_token = True

        self._preprocess()

    def _tokenize(self, text, id):

        out = self.tokenizer(text,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length)

        out[id + '_input_ids'] = out.pop('input_ids')
        out[id + '_attention_mask'] = out.pop('attention_mask')

        return out

    def _preprocess(self):
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent0'], 'sent0'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent1'], 'sent1'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['hard_neg'], 'hard_neg'), batched=True)
        self.ds.set_format(
            type="torch",
            columns=["sent0_input_ids", "sent0_attention_mask",
                     "sent1_input_ids", "sent1_attention_mask",
                     "hard_neg_input_ids", "hard_neg_attention_mask",]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Use first N rows for a pilot dataset")
    parser.add_argument("--tokenizer_path", type=str, default="../pretrained/sarvam-1/",
                        help="Path to tokenizer (default: ../pretrained/sarvam-1/)")
    parser.add_argument("--max_length", type=int, default=150,
                        help="Max token length for padding/truncation (default: 150)")
    parser.add_argument("--input_csv", type=str, default="nli_for_simcse.csv",
                        help="Input CSV file (default: nli_for_simcse.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-derived from input_csv)")
    args = parser.parse_args()

    nlip = NLIPreprocess(args.input_csv,
                         tokenizer_path=args.tokenizer_path,
                         max_length=args.max_length,
                         num_rows=args.num_rows)

    if args.output_dir:
        out_dir = args.output_dir
    elif args.num_rows:
        out_dir = "./processed_pilot/"
    else:
        out_dir = "./processed/"

    nlip.ds.save_to_disk(out_dir)
    print(f"Saved {len(nlip.ds)} examples to {out_dir}")
