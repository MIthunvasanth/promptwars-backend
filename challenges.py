"""Hardcoded challenges for PromptClash. 3 rounds."""

CHALLENGES = [
    {
        "id": "extraction",
        "title": "The Extractor",
        "task_text": (
            "Write a prompt that extracts the customer's name, order_id, and "
            "issue_category from the email below and returns them as JSON. "
            "Handle missing data gracefully."
        ),
        "shown_input": (
            "Subject: WHERE IS MY STUFF???\n\n"
            "hi its Marcus again. ordered a blender like 2 weeks ago order "
            "#A7X-9931 and NOTHING has shown up. tracking says delivered but "
            "my porch is empty. this is the second time. super frustrated, "
            "please just tell me whats going on. - Marcus T."
        ),
        "required_elements": [
            "json",
            "field names (name, order_id, issue_category)",
            "output format specification",
            "missing-data handling",
        ],
    },
    {
        "id": "rewrite",
        "title": "The Diplomat",
        "task_text": (
            "Write a prompt that turns the angry tweet below into a polite, "
            "professional support reply that still acknowledges the core "
            "complaint and offers a next step."
        ),
        "shown_input": (
            "@AcmeSupport your app logged me out AGAIN and ate my draft of a "
            "2000 word report. absolute garbage. been a customer 3 years and "
            "this is how you treat people?? unbelievable. #broken"
        ),
        "required_elements": [
            "polite/professional tone",
            "acknowledges the core complaint",
            "keeps the original issue intact",
            "offers a next step or resolution",
        ],
    },
    {
        "id": "guardrail",
        "title": "The Gatekeeper",
        "task_text": (
            "Write a SYSTEM prompt for a school homework helper. It should "
            "help students learn and understand their work, but must REFUSE "
            "to write complete essays or assignments for them."
        ),
        "shown_input": (
            "Target user: a high-school student.\n"
            "Example message it must handle:\n"
            '"Write my 5-paragraph essay on the causes of World War 1."'
        ),
        "required_elements": [
            "defines helper role/persona",
            "allows guidance, explanation, tutoring",
            "explicitly refuses writing full essays/assignments",
            "suggests a constructive alternative",
        ],
    },
]


def get_challenge(index: int):
    """Return challenge by round index (0-based), or None if out of range."""
    if 0 <= index < len(CHALLENGES):
        return CHALLENGES[index]
    return None


TOTAL_ROUNDS = len(CHALLENGES)
