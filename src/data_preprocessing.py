"""Clean raw Enron email data and build incoming/reply pairs."""

import re
from pathlib import Path
from email import message_from_string
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

def parse_email(raw):
    msg = message_from_string(raw)
    body = msg.get_payload()
    if isinstance(body, list):
        body = body[0].get_payload()
    return {
        'message_id': msg.get('Message-ID', '').strip(),
        'in_reply_to': msg.get('In-Reply-To', '').strip(),
        'body': body
    }

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'>.*?\n', '', text)        # quoted lines
    text = re.sub(r'[-_]{2,}.*', '', text, flags=re.DOTALL)  # signatures
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_pairs(df):
    print("Parsing emails...")
    email_dict = {}
    for _, row in df.iterrows():
        try:
            parsed = parse_email(str(row['message']))
            if parsed['message_id']:
                email_dict[parsed['message_id']] = parsed
        except:
            continue

    print(f"Parsed {len(email_dict)} emails. Building pairs...")
    pairs = []
    for msg_id, email in email_dict.items():
        irt = email['in_reply_to']
        if irt and irt in email_dict:
            incoming = clean_text(email_dict[irt]['body'])
            reply = clean_text(email['body'])
            if (10 < len(incoming) < 2000) and (10 < len(reply) < 500):
                pairs.append({
                    'incoming_email': incoming,
                    'reply': reply
                })
    return pd.DataFrame(pairs)

def main():
    print("Loading emails.csv ...")
    df = pd.read_csv(DATA_RAW / "emails.csv")
    print(f"Total emails: {len(df)}")

    pairs_df = build_pairs(df)
    pairs_df = pairs_df.drop_duplicates().reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "email_pairs.csv"
    pairs_df.to_csv(out, index=False)
    print(f"Saved {len(pairs_df)} pairs to {out}")

if __name__ == "__main__":
    main()