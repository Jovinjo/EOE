import os
import re
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class SynonymFetcher:
    def __init__(self):
        self.dataset_info = {
            'cub100_ID': {'class_type': 'bird', 'num_classes': 100},
            'pet18_ID': {'class_type': 'pet (including dogs and cats)', 'num_classes': 18},
        }

    def fetch(self, name: str, in_dataset: str) -> str:
        """
        Fetch a synonym, scientific name, or name in another language for the given class label,
        using dataset metadata to determine class_type.
        """
        class_type = self.dataset_info[in_dataset]["class_type"]

        system_prompt = (
            f"You are a fine-grained classification assistant. "
            f"When given a {class_type} species name, reply with *exactly one* known synonym, scientific name, "
            f"or name in another language. If no widely used alternative exists, reply with the original label. "
            f"Do not return *none*. Only return the name. Do not include explanations, punctuation, or prefixes.\n\n"
            "Here are a few examples:\n"
            "Input: American Goldfinch → Output: Eastern Goldfinch\n"
            "Input: Indigo Bunting → Output: Passerina cyanea\n"
            "Input: Bengal Cat → Output: Leopardette\n"
            "Input: Gadwall → Output: Anas strepera\n"
        )

        user_prompt = f"Class label: {name}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            raw = response.choices[0].message.content.strip()

            # Clean up any unnecessary formatting
            if ":" in raw:
                raw = raw.split(":")[-1].strip()
            if "known as" in raw:
                raw = re.split(r"known as", raw, flags=re.IGNORECASE)[-1].strip()

            return raw.rstrip(".")
        except Exception as e:
            print(f"[ERROR] Failed to fetch synonym for '{name}': {e}")
            return name