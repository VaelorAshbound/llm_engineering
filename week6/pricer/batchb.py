"""Groq batch orchestration. Same flow as batch.py, but the prompt / model /
message construction is imported from preprocessorb (single source of truth)
rather than re-declared here.
"""

import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pricer.preprocessorb import MODEL, REASONING_EFFORT, messages_for
from tqdm.notebook import tqdm

load_dotenv(override=True)
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")


# --- Note: self vs cls (instance methods vs @classmethod) ---------------------
# This class is a good worked example of the distinction:
#
#   * Instance fields are tied to one object: self.items, self.start, self.end,
#     self.file_id, etc. are set per-Batch in __init__ and differ per batch.
#     Methods that need a SPECIFIC batch's data take `self` (make_file,
#     send_file, submit_batch, is_ready, fetch_output, apply_output).
#
#   * Class fields are tied to the class and shared by all instances:
#     BATCH_SIZE and `batches` below. There is ONE `batches` list for the whole
#     class. Methods that don't need a specific batch — they build batches or
#     operate on the shared list — take `cls` and are marked @classmethod
#     (create, run, fetch, save, load). `create` is an alternative constructor:
#     it builds Batch objects, so no instance exists yet -> classmethod, not self.
#
#   Rule of thumb: needs a specific object's data -> normal method (self);
#   builds an object or only touches class-level/shared state -> @classmethod (cls).
#
#   Gotcha: a class field changes for ALL instances only when set on the class
#   (cls.batches = ... or Batch.batches = ...). Setting it on one instance
#   (some_batch.batches = ...) silently creates a per-object field that shadows
#   the shared one instead of updating it.
# ------------------------------------------------------------------------------
class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches_folder = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches_folder.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": messages_for(item.full),
                "reasoning_effort": REASONING_EFFORT,
            },
        }
        return json.dumps(line)

    def make_file(self):
        batch_file = self.batches_folder / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches_folder / self.filename
        with batch_file.open("rb") as f:
            response = groq.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        assert self.file_id is not None, "file_id is not set; call send_file() first"
        response = groq.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        assert self.batch_id is not None, (
            "batch_id is not set; call submit_batch() first"
        )
        response = groq.batches.retrieve(self.batch_id)
        if response.status == "completed":
            self.output_file_id = response.output_file_id
        return response.status == "completed"

    def fetch_output(self):
        assert self.output_file_id is not None, (
            "output_file_id is not set; batch not yet completed"
        )
        output_file = str(self.output / self.filename)
        groq.files.content(self.output_file_id).write_to_file(output_file)

    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            cls.batches.append(cls(items, start, end, lite))
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if not batch.done and batch.is_ready():
                batch.fetch_output()
                batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
