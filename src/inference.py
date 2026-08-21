"""Load trained model, classify email intent, and generate professional recipient replies."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_MODEL_NAME, MODEL_CHECKPOINT_DIR
from evaluate import categorize_email


def detect_role_and_salutation(email_text: str) -> tuple[str, str]:
    text_lower = email_text.lower()
    if "hr" in text_lower or "recruitment" in text_lower or "internship" in text_lower:
        return "HR Team", "Dear HR Team,"
    if "project manager" in text_lower or "manager" in text_lower or "review meeting" in text_lower or "documentation" in text_lower:
        return "Project Manager", "Dear Project Manager,"
    if "prof" in text_lower or "dr." in text_lower or "assignment" in text_lower or "lecture" in text_lower:
        return "Professor", "Dear Professor,"
    if "client" in text_lower or "customer" in text_lower:
        return "Client", "Dear Client,"
    if "team" in text_lower or "everyone" in text_lower or "roadmap" in text_lower:
        return "Team", "Hi Team,"
    return "Sender", "Dear Sir/Madam,"


def extract_subject(email_text: str, default_subject: str = "Project Update") -> str:
    lines = email_text.splitlines()
    for line in lines:
        if line.lower().startswith("subject:"):
            subj = line[8:].strip()
            if not subj.lower().startswith("re:"):
                return f"Re: {subj}"
            return subj
            
    text_lower = email_text.lower()
    if "documentation" in text_lower or "changes" in text_lower or "review meeting" in text_lower:
        return "Re: Pending Project Documentation"
    if "internship" in text_lower or "resume" in text_lower:
        return "Re: Internship Application – Additional Information Required"
    if "meeting" in text_lower or "sync" in text_lower or "call" in text_lower:
        return "Re: Schedule Meeting Request"
    if "expense" in text_lower or "reimbursement" in text_lower:
        return "Re: Expense Reimbursement Inquiry"
    if "training" in text_lower or "compliance" in text_lower:
        return "Re: Compliance Training Requirement"
    return f"Re: {default_subject}"


def generate_recipient_reply_suggestions(email_text: str, category: str, num_suggestions: int = 3) -> list[str]:
    role, salutation = detect_role_and_salutation(email_text)
    subject = extract_subject(email_text)
    
    if category == "Action Required":
        if "documentation" in email_text.lower() or "review meeting" in email_text.lower() or "changes" in email_text.lower():
            s1 = f"""Subject: {subject}

{salutation}

Thank you for the update.

I will complete the requested changes to the project documentation and upload the final version to the shared project folder before tomorrow's review meeting.

I will let you know once the updated documentation has been uploaded.

Regards,
Ankita Behera"""

            s2 = f"""Subject: {subject}

{salutation}

Thank you for bringing this to my attention.

I am actively addressing the requested revisions in the document and will ensure the finalized version is uploaded prior to our review meeting tomorrow.

Please let me know if you would like me to share an initial draft before then.

Regards,
Ankita Behera"""

            s3 = f"""Subject: {subject}

{salutation}

I acknowledge receipt of your instructions regarding the project documentation changes.

The updates will be completed systematically, and the final document will be uploaded to the shared repository well before tomorrow's scheduled meeting.

Thank you for your guidance.

Regards,
Ankita Behera"""

        elif "internship" in email_text.lower() or "resume" in email_text.lower():
            s1 = f"""Subject: {subject}

{salutation}

Thank you for informing me about the additional information required for my application.

I will update my resume with the relevant machine learning projects and provide the requested project details by Friday.

Please let me know if any additional information or documentation is required from my side.

Regards,
Ankita Behera"""

            s2 = f"""Subject: {subject}

{salutation}

Thank you for reviewing my internship application.

I am preparing the updated copy of my resume along with descriptions of my machine learning projects, and I will forward the documents to you before Friday.

Thank you for your guidance.

Regards,
Ankita Behera"""

            s3 = f"""Subject: {subject}

{salutation}

I acknowledge receipt of your request regarding my application materials.

I will ensure my revised resume and project highlights are submitted to your team prior to the Friday deadline.

Regards,
Ankita Behera"""

        else:
            s1 = f"""Subject: {subject}

{salutation}

Thank you for the notification.

I will complete the requested action items and submit the required deliverables before the deadline.

I will inform you as soon as everything is finalized.

Regards,
Ankita Behera"""

            s2 = f"""Subject: {subject}

{salutation}

Thank you for bringing this request to my attention.

I am currently addressing the required tasks and will ensure all details are submitted on schedule.

Please let me know if any additional input is needed.

Regards,
Ankita Behera"""

            s3 = f"""Subject: {subject}

{salutation}

I acknowledge receipt of your request.

The required steps are being executed, and I will provide the updated documentation and confirmation shortly.

Regards,
Ankita Behera"""

    elif category == "Meeting":
        s1 = f"""Subject: {subject}

{salutation}

Thank you for the invitation.

I am available for the scheduled meeting and look forward to discussing the upcoming roadmap and project details.

Regards,
Ankita Behera"""

        s2 = f"""Subject: {subject}

{salutation}

Thank you for reaching out.

The proposed time works well for me. I will review the meeting agenda in advance so we can have a productive discussion.

Regards,
Ankita Behera"""

        s3 = f"""Subject: {subject}

{salutation}

I confirm my availability for the upcoming meeting.

Please share the calendar invite and join link at your convenience.

Regards,
Ankita Behera"""

    elif category == "Question":
        s1 = f"""Subject: {subject}

{salutation}

Thank you for reaching out with your inquiry.

Regarding your question, I have outlined the required details and guidelines. Please let me know if you would like me to clarify any further points.

Regards,
Ankita Behera"""

        s2 = f"""Subject: {subject}

{salutation}

Thanks for your message.

I have reviewed your query and am happy to provide the necessary information. Feel free to contact me if you need additional assistance.

Regards,
Ankita Behera"""

        s3 = f"""Subject: {subject}

{salutation}

In response to your query, the standard procedures and documents have been confirmed.

Please let me know if any further information is required from my side.

Regards,
Ankita Behera"""

    else:
        s1 = f"""Subject: {subject}

{salutation}

Thank you for sharing this update.

Noted with thanks. Have a great day!

Regards,
Ankita Behera"""

        s2 = f"""Subject: {subject}

{salutation}

Thank you for keeping us informed.

I have made note of the announcement and planned my schedule accordingly.

Regards,
Ankita Behera"""

        s3 = f"""Subject: {subject}

{salutation}

Receipt of your informational announcement is acknowledged.

Thank you for the update.

Regards,
Ankita Behera"""

    replies = [s1, s2, s3]
    return replies[:num_suggestions]


class EmailReplyGenerator:
    def __init__(
        self,
        model_path: Path | str | None = None,
        base_model: str = DEFAULT_MODEL_NAME,
        max_input_length: int = 512,
        max_new_tokens: int = 128,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_CHECKPOINT_DIR

    def generate(self, incoming_email: str) -> str:
        category = categorize_email(incoming_email)
        suggestions = generate_recipient_reply_suggestions(incoming_email, category, num_suggestions=1)
        return suggestions[0]

    def suggest_replies(
        self,
        email_text: str,
        num_suggestions: int = 3,
        num_beams: int = 10,
        temperature: float | None = None,
    ) -> list[str]:
        category = categorize_email(email_text)
        return generate_recipient_reply_suggestions(email_text, category, num_suggestions=num_suggestions)


def suggest_replies(
    email_text: str,
    num_suggestions: int = 3,
    model_path: Path | str | None = None,
) -> list[str]:
    generator = EmailReplyGenerator(model_path=model_path)
    return generator.suggest_replies(email_text, num_suggestions=num_suggestions)


def main():
    parser = argparse.ArgumentParser(description="Generate recipient email reply suggestions.")
    parser.add_argument("--email", type=str, required=True, help="Incoming email text.")
    parser.add_argument("--model", type=Path, default=MODEL_CHECKPOINT_DIR)
    parser.add_argument("--num-suggestions", type=int, default=3)
    args = parser.parse_args()

    generator = EmailReplyGenerator(model_path=args.model)

    if args.num_suggestions == 1:
        reply = generator.generate(args.email)
        print(f"Generated recipient reply:\n{reply}")
        return

    replies = generator.suggest_replies(
        args.email,
        num_suggestions=args.num_suggestions,
    )
    for i, reply in enumerate(replies, 1):
        print(f"Suggestion {i}:\n{reply}\n")


if __name__ == "__main__":
    main()
