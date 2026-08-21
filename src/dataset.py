"""PyTorch Dataset for email reply generation."""

from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, df, tokenizer, max_input=512, max_target=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = "reply to email: " + str(self.df.loc[idx, "incoming_email"])
        target_text = str(self.df.loc[idx, "reply"])

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }
