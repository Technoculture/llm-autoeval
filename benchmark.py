"""
This script implement a factory pattern to implement any benchmark for evaluation.
"""

import os
import json
import random
import pandas as pd
import dspy
import logging

from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk

from medprompt import DefaultModule

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT_PROMPTS = {
    "mmlu_general": "medmcqa",
    "mmlu_medical": "medmcqa",
    "medqa": "medqa",
    "medmcqa": "medmcqa",
    "pubmedqa": "pubmedqa",
}


def benchmark_factory(name):
    """
    Creates a benchmark object.

    :param name: str, with the benchmark name.
    return:
    """
    # Note: benchmark is instantiated *after* selection.
    # pubmedqa, open_pubmedqa, medqa  datsets not loading.
    # Done: medmcqa, medqa4, arc,
    # Left: medication, truthful, mmlu, gsm8k, blurb, hellaswag, winogrande
    factories = {
        "medmcqa": Medmcqa,
        "pubmedqa": ClosedPubMedQA,
        "open_pubmedqa": PubMedQA,
        "medqa": MedQA,
        "medqa4": Medqa4,
        "medicationqa": MedicationQA,
        "mmlu_medical": MMLU,
        "arc": ARC,
        "hellaswag": HellaSwag,
        "winogrande": Winogrande,
        "blurb": Blurb,
        "truthfulqa": TruthfulQA,
        "gsm8k": GSM8K,
        "mmlu_general": MMLU,
    }
    if name not in factories:
        raise ValueError(
            "Benchmark {} not found. \
                         Select one of the following: {}".format(
                name, list(factories.keys())
            )
        )
    return factories[name](name)


def load_instruction(prompt_name):
    """
    Loads the instruction for the given benchmark.

    :param benchmark: str, the name of the benchmark
    :param prompt_name: str, the name of the prompt to be used
    """
    path = os.path.join(ROOT_DIR, "evaluation", "instructions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Please save the different prompts to instructions.json"
        )

    with open(path) as f:
        prompts = json.load(f)
    return prompts[prompt_name]


class Benchmark:
    def __init__(self, name):
        """
        Class to implement a benchmark for evaluation.

        :param name: str, with the benchmark name.
        :param path: str (optional), the path to the benchmark data.
        :param splits: list of str, the splits of the data: train / test
        :param hub_name: str, the name of the HuggingFace hub dataset.
        :param dir_name: str, the name of the directory where the data is stored.
        :param train_data: HuggingFace Dataset, the train data.
        :param test_data: HuggingFace Dataset, the test data.
        :param generations: HuggingFace Dataset, the generations.
        :param subsets: list of str (optional), the subsets of the data to download from the HuggingFace hub.
        :param has_instruction: bool, whether the dataset already contains instructions.
        :param local_path: str (optional), the path to a directory holding train and test json local data files.
        """
        self.name = name
        self.path = None
        self.splits = None
        self.hub_name = None
        self.dir_name = None
        self.train_data = None
        self.test_data = None
        self.generations = None
        self.subsets = None
        self.has_instructions = False
        self.local_path = None

    def load_from_local(self):
        """
        Downloads the benchmark data from local files (for 1st time loading).
        """
        print(
            f"Downloading {self.name} benchmark from local directory {self.local_path}."
        )
        if not os.path.exists(self.local_path):
            raise ValueError(
                f"Local path {self.local_path} does not exist. \
                             Please provide a valid local_path argument to load_data(). \
                             This directory should contain train and test json files."
            )
        paths = [
            os.path.join(self.local_path, file) for file in os.listdir(self.local_path)
        ]
        train_paths = [path for path in paths if "train" in path]
        test_paths = [path for path in paths if "test" in path or "val" in path]
        if len(train_paths) == 0:
            raise ValueError(
                "Could not find a train file in the local directory. \
                                Please add a file with 'train' in its name."
            )
        if len(test_paths) == 0:
            raise ValueError(
                "Could not find a test file in the local directory. \
                                Please add a file with 'test' or 'val' in its name."
            )
        if len(train_paths) > 1:
            print(
                "Multiple train files found. Using the first one: {}".format(
                    train_paths[0]
                )
            )
        if len(test_paths) > 1:
            print(
                "Multiple test files found. Using the first one: {}".format(
                    test_paths[0]
                )
            )
        data_files = {"train": train_paths[0], "test": test_paths[0]}
        dataset = load_dataset(
            "json", data_files=data_files, download_mode="force_redownload"
        )
        dataset.save_to_disk(self.path)

    def load_from_hub(self):
        """
        Downloads the benchmark data from the HuggingFace hub (for 1st time loading)
        This is specific to each benchmark and must be implemented in the extended class.
        """
        print(f"Downloading benchmark from HuggingFace hub ({self.hub_name}).")
        try:
            if self.subsets is None:
                load_dataset(
                    self.hub_name,
                    cache_dir=os.path.join(ROOT_DIR, "benchmarks", "datasets"),
                    download_mode="force_redownload",
                )
            else:
                for subset in self.subsets:
                    load_dataset(
                        self.hub_name,
                        subset,
                        cache_dir=os.path.join(ROOT_DIR, "benchmarks", "datasets"),
                        download_mode="force_redownload",
                    )
        except Exception as e:
            print(e)
            raise ValueError(
                "Default Huggingface loader failed for benchmark {}. \
                             Try implementing a custom load_from_hub function.".format(
                    self.name
                )
            )

    def load_data(self, partition="train", local_path=None):
        """
        Loads benchmark data from a local directory, or from the HuggingFace hub if not yet downloaded.
        Based on the input partition type, instantiates the respective class attribute.

        :param path: str (optional), the path to the benchmark data.
        :param partition: str, the split of the data: train / test
        :param local_path: str (optional), the path to a directory holding train and test json local data files.
        """
        print("=" * 50 + f"\nLoading data for benchmark {self.name}.\n")
        if self.name == "hellaswag":
            if partition == "train":
                self.train_data = load_dataset("Rowan/hellaswag", split=partition)
            elif partition in ["test", "validation"]:
                self.test_data = load_dataset("Rowan/hellaswag", split=partition)
        else:
            if partition not in self.splits:
                raise ValueError(
                    "Please provide a valid partition split: {}".format(self.splits)
                )
            if local_path:
                self.local_path = local_path
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                if self.local_path:
                    self.load_from_local()
                else:
                    self.load_from_hub()
            try:
                if self.local_path:
                    dataset = load_from_disk(self.path)
                    if partition == "train":
                        self.train_data = dataset["train"]
                    elif partition in ["test", "validation"]:
                        self.test_data = dataset[partition]
                else:
                    if self.subsets is None:
                        if partition == "train":
                            self.train_data = load_dataset(self.path, split=partition)
                        elif partition in ["test", "validation"]:
                            self.test_data = load_dataset(self.path, split=partition)
                    else:
                        if partition == "train":
                            self.train_data = aggregate_datasets(
                                self.path, self.subsets, partition=partition
                            )
                        elif partition in ["test", "validation"]:
                            self.test_data = aggregate_datasets(
                                self.path, self.subsets, partition=partition
                            )

            except ValueError as e:
                print(e)
                raise ValueError(
                    "Couldn't load benchmark {} from local path.".format(self.name)
                )

    def save_data(self, partition="train"):
        """
        Saves any preprocessing data partition.

        :param data: pd.DataFrame
        :param file_name: str
        """
        path = os.path.join("benchmarks", "preprocessing", f"{self.name}_{partition}")
        print("Saving {} data to the following path: {}".format(self.name, path))
        if partition == "train":
            pd.to_pickle(self.train_data, path)
        elif partition == "test":
            pd.to_pickle(self.test_data, path)

    def preprocessing(self, partition="train"):
        """
        Applies a custom pre-processing over the partition.
        If instruction is provided, preprends it to the question
        Updates the train or test self attributes.

        :param _preprocess: function: dict -> dict, the preprocessing function to apply.
        :param partition: str, the split of the data: train / test
        """
        try:
            if partition == "train":
                self.train_data = self.train_data.map(self.custom_preprocessing)
            elif partition in ["test", "validation"]:
                self.test_data = self.test_data.map(self.custom_preprocessing)
            else:
                raise ValueError(
                    "Please provide a valid partition split: train or test"
                )
        except Exception as e:
            print(e)
            raise ValueError(
                "Error when pre-processing {} {} data.".format(self.name, partition)
            )

    def custom_preprocessing(self):
        """
        Wraps a pre-processing function (dict -> dict) specific to the benchmark.
        Needs to be overriden in the extended class.

        The return dictionary must contains keys 'prompt' & 'answer' for inference to work.
        """
        raise NotImplementedError("Implement custom_preprocessing() in a child class.")

    def add_instruction(self, instruction=None, cot_column=None, partition="train"):
        """
        Adds instructions to the data based on the input partition.

        :param instruction: dict, with the `system` and `user` instructions. If None, then it creates prompt with few shot
        :param cot_column: str, the column that has the CoT explanation behind the gold answer.
        :param partition: str, the split of the data: train / test
        """

        def _add_instruction(row):
            row["prompt"] = "{}\n{}\n{}\n".format(
                instruction["system"], row["prompt"], instruction["user"]
            )
            if cot_column:
                row["gold"] = "{}.\nThe answer is: {} ###".format(
                    row[cot_column], row["gold"]
                )
            return row

        if partition == "train":
            self.train_data = self.train_data.map(_add_instruction)
        elif partition == "test" or partition == "validation":
            self.test_data = self.test_data.map(_add_instruction)
        else:
            raise ValueError(
                "Please provide a valid partition split: {}".format(self.splits)
            )

    def add_few_shot(self, shots=8, seed=42, load_cot=False):
        def _get_question(prompt):
            if "Question:" in prompt:
                return prompt.split("Question:")[1]
            else:
                return prompt

        if load_cot:
            assert self.name in COT_PROMPTS, "No CoT prompts found for {}.".format(
                self.name
            )
            cot_path = os.path.join(
                ROOT_DIR, "evaluation", "prompt_cot", f"{COT_PROMPTS[self.name]}.jsonl"
            )
            samples = pd.read_json(cot_path, lines=True).to_dict(orient="records")
            demonstrations = random.sample(samples, shots)
            few_shot_prompt = "\n\n".join(
                [
                    f"Question: {_get_question(demo['prompt'])}\n{demo['gold']}"
                    for demo in demonstrations
                ]
            )
        else:
            assert self.train_data is not None, "Please load the train data first."
            demonstrations = self.train_data.shuffle(seed=seed).select(range(shots))
            few_shot_prompt = "\n\n".join(
                [
                    "{}\nThe answer is: {}".format(demo["prompt"], demo["gold"])
                    for demo in demonstrations
                ]
            )

        def _add_few_shot(row):
            row["prompt"] = "{}\n\n{}".format(few_shot_prompt, row["prompt"])
            return row

        self.test_data = self.test_data.map(_add_few_shot)

    def add_generations(self, data):
        """
        Adds the generations to the respective class attribute as a HuggingFace Dataset.

        :param data: pd.DataFrame or HuggingFace Dataset
        """
        if isinstance(data, pd.DataFrame):
            self.generations = Dataset.from_pandas(data)
        elif isinstance(data, Dataset):
            self.generations = data

    def save_generations(self, checkpoint_name, shots=0):
        """
        Saves the generations in the respective directory.
        """
        path = os.path.join(ROOT_DIR, "benchmarks", "generations")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if shots == 0:
            gen_path = os.path.join(path, f"{self.name}-{checkpoint_name}.jsonl")
        else:
            gen_path = os.path.join(
                path, f"{self.name}-{checkpoint_name}-{str(shots)}-shot.jsonl"
            )

        self.generations.to_json(gen_path, orient="records")
        print(
            "Stored {} generations to the following path: {}".format(
                self.name, gen_path
            )
        )

    def load_generations(self, checkpoint_name):
        """
        Loads the generations from the respective directory.
        """
        path = os.path.join(
            ROOT_DIR, "benchmarks", "generations", f"{self.name}_{checkpoint_name}.json"
        )
        if not os.path.exists(path):
            raise ValueError(
                "No generations found for {} at path: {}. \
                             Please run inference first.".format(self.name, path)
            )
        print(
            "Loading {} generations from the following path: {}".format(self.name, path)
        )
        self.generations = pd.read_json(path)


class Medmcqa(Benchmark):
    """
    MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset
    designed to address real-world medical entrance exam questions.

    Huggingface card: https://huggingface.co/datasets/medmcqa
    """

    def __init__(self, name="medmcqa") -> None:
        super().__init__(name)
        self.hub_name = "medmcqa"
        self.dir_name = "medmcqa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        row["optionsKey"] = "A. {} B. {} C. {} D. {}".format(
            row["opa"], row["opb"], row["opc"], row["opd"]
        )
        row["prompt"] = format_mcq(row["question"], row["optionsKey"])
        answer = int(row["cop"])
        row["gold"] = chr(ord("A") + answer) if answer in [0, 1, 2, 3] else None
        return row


class PubMedQA(Benchmark):
    """
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    """

    def __init__(self, name="pubmedqa") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = "bigbio___pubmed_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["pubmed_qa_labeled_fold0_source"]
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):
        row["prompt"] = row["QUESTION"]  #  f"{row['CONTEXTS'][0]}\n{row['QUESTION']}"
        row["gold"] = row["final_decision"]
        row["long_answer"] = row["LONG_ANSWER"]
        return row


class ClosedPubMedQA(Benchmark):
    """
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    """

    def __init__(self, name="pubmedqa") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = "bigbio___pubmed_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["pubmed_qa_labeled_fold0_source"]
        self.num_options = 3

    @staticmethod
    def custom_preprocessing(row):
        context = "\n".join(row["CONTEXTS"])
        row["prompt"] = f"{context}\n{row['QUESTION']}"
        row["gold"] = row["final_decision"]
        row["long_answer"] = row["LONG_ANSWER"]
        return row


class PubMedQAValidation(Benchmark):
    """
    PubMedQA is a novel biomedical question answering (QA) dataset.
    Its task is to answer research biomedical questions with yes/no/maybe using PubMed abstracts.

    Huggingface card: https://huggingface.co/datasets/bigbio/pubmed_qa
    """

    def __init__(self, name="pubmedqa") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/pubmed_qa"
        self.dir_name = "bigbio___pubmed_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["validation"]
        # self.subsets = ['pubmed_qa_labeled_fold1_bigbio_qa'] + ['pubmed_qa_artificial_source']
        self.num_options = 3
        self.local_path = os.path.join(
            ROOT_DIR, "benchmarks", "datasets", "pubmedqa_pubmedqa_validation.jsonl"
        )

    @staticmethod
    def custom_preprocessing(row):
        row["prompt"] = row["QUESTION"]
        row["gold"] = row["final_decision"]
        return row


class MedQA(Benchmark):
    """
    MedQA is a dataset for solving medical problems collected from the professional medical board exams.

    Huggingface card: https://huggingface.co/datasets/bigbio/med_qa
    """

    def __init__(self, name="medqa") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/med_qa"
        self.dir_name = "bigbio___med_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.num_options = 5

    @staticmethod
    def custom_preprocessing(row):
        choices = [opt["value"] for opt in row["options"]]
        row["prompt"] = format_mcq(row["question"], choices)
        for opt in row["options"]:
            if opt["value"] == row["answer"]:
                row["gold"] = opt["key"]
                break
        return row


class Medqa4(Benchmark):
    """
    MedQA is a dataset for solving medical problems collected from the professional medical board exams.

    Huggingface card: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    """

    def __init__(self, name="medqa4") -> None:
        super().__init__(name)
        self.hub_name = "GBaker/MedQA-USMLE-4-options"
        self.dir_name = "GBaker___med_qa-usmle-4-options"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "test"]
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        row["optionsKey"] = "A. {} B. {} C. {} D. {}".format(
            row["options"]["A"],
            row["options"]["B"],
            row["options"]["C"],
            row["options"]["D"],
        )
        row["prompt"] = format_mcq(row["question"], row["optionsKey"])
        row["gold"] = row["answer_idx"]

        return row


class MedicationQA(Benchmark):
    """
    MedicationQA is a dataset of consumer health questions about medications.
    Huggingface card: https://huggingface.co/datasets/truehealth/medicationqa
    """

    def __init__(self, name="medicationqa") -> None:
        super().__init__(name)
        self.hub_name = "truehealth/medicationqa"
        self.dir_name = "truehealth___parquet"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train"]

    @staticmethod
    def custom_preprocessing(row):
        """
        Wraps a pre-processing function (dict -> dict) specific to the benchmark.
        Probably will need to be overriden in the extended class.
        """
        row["prompt"] = (row["Question"],)
        row["gold"] = row["Answer"]
        return row


class TruthfulQA(Benchmark):
    """
    TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions
    Huggingface card: https://huggingface.co/datasets/truthful_qa
    """

    def __init__(self, name="truthfulqa") -> None:
        super().__init__(name)
        self.hub_name = "truthful_qa"
        self.dir_name = "truthful_qa"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["validation"]
        self.subsets = ["multiple_choice"]
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        options = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold_option = options[labels.index(1)]
        options.remove(gold_option)
        wrong_options = random.choices(options, k=3)
        choices = [gold_option] + wrong_options
        random.shuffle(choices)
        gold_id = choices.index(gold_option)
        row["prompt"] = format_mcq(row["question"], choices)
        row["gold"] = ["A", "B", "C", "D"][gold_id]
        return row


class MMLU(Benchmark):
    """
    Measuring Massive Multitask Language Understanding
    Huggingface card: https://huggingface.co/datasets/lukaemon/mmlu
    """

    def __init__(self, name) -> None:
        super().__init__(name)
        self.hub_name = "lukaemon/mmlu"
        self.dir_name = "lukaemon___mmlu"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets_general = [
            "college_computer_science",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_statistics",
            "machine_learning",
            "global_facts",
        ]
        self.subsets_medical = [
            "anatomy",
            "college_biology",
            "college_medicine",
            "professional_medicine",
            "medical_genetics",
            "virology",
            "clinical_knowledge",
            "high_school_biology",
            "high_school_chemistry",
            "nutrition",
            "college_chemistry",
        ]
        self.subsets = self.subsets_medical
        if name == "mmlu_general":
            self.subsets = self.subsets_general
        self.num_options = 4

    @staticmethod
    def custom_preprocessing(row):
        options = [row["A"], row["B"], row["C"], row["D"]]
        row["optionsKey"] = " ".join(
            [
                "{}. {}".format(label, text)
                for label, text in zip(
                    ["A", "B", "C", "D"], [row["A"], row["B"], row["C"], row["D"]]
                )
            ]
        )
        row["prompt"] = format_mcq(row["input"], options)
        row["gold"] = row["target"]
        row["subset"] = row["subset"]
        return row


class GSM8K(Benchmark):
    """
    GSM8K (Grade School Math 8K) is a dataset of 8.5K grade school math word problems.
    Huggingface card: https://huggingface.co/datasets/gsm8k
    """

    def __init__(self, name="gsm8k") -> None:
        super().__init__(name)
        self.hub_name = "gsm8k"
        self.dir_name = "gsm8k"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "test"]
        self.subsets = ["main"]

    @staticmethod
    def custom_preprocessing(row):
        row["prompt"] = row["question"]
        row["gold"] = row["answer"].split("####")[1].strip()
        row["steps"] = row["answer"].split("####")[0].strip()
        return row


class Blurb(Benchmark):
    """
    BLURB is a collection of resources for biomedical natural language processing.
    Huggingface card: https://huggingface.co/datasets/bigbio/blurb
    """

    def __init__(self, name="blurb") -> None:
        super().__init__(name)
        self.hub_name = "bigbio/blurb"
        self.dir_name = "bigbio___blurb"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["bc2gm", "bc5chem", "bc5disease", "jnlpba", "ncbi_disease"]

    @staticmethod
    def custom_preprocessing(row):
        tokens = row["tokens"]
        tags = row["ner_tags"]
        entity_type = row["type"]

        instruction = f"Given the following sentence, tell me which part of this sentence is a {entity_type} expression. There may be multiple expressions in this sentence."
        prompt = f"{instruction}\n\n Sentence: {' '.join(tokens)}"
        row["prompt"] = prompt
        row["gold"] = "#".join(Blurb.get_entities(tokens, tags))
        return row

    @staticmethod
    def get_entities(tokens, tags):
        entities = []
        entity = []
        for token, tag in zip(tokens, tags):
            if tag == 1:
                if entity:
                    entities.append(" ".join(entity))
                entity = [token]
            elif tag == 2:
                entity.append(token)
            elif tag == 0 and entity:
                entities.append(" ".join(entity))
                entity = []
        if entity:
            entities.append(" ".join(entity))
        return entities


class ARC(Benchmark):
    """
    Huggingface card: https://huggingface.co/datasets/ai2_arc
    """

    def __init__(self, name="arc") -> None:
        super().__init__(name)
        self.hub_name = "ai2_arc"
        self.dir_name = "ai2_arc"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]
        self.subsets = ["ARC-Challenge"]

    @staticmethod
    def custom_preprocessing(row):
        row["optionsKey"] = " ".join(
            [
                "{}. {}".format(label, text)
                for label, text in zip(row["choices"]["label"], row["choices"]["text"])
            ]
        )
        row["prompt"] = format_mcq(row["question"], row["optionsKey"])
        row["gold"] = row["answerKey"]
        return row


class HellaSwag(Benchmark):

    """
    Huggingface card: https://huggingface.co/datasets/Rowan/hellaswag
    """

    def __init__(self, name="hellaswag") -> None:
        super().__init__(name)
        self.hub_name = "Rowan/hellaswag"
        self.dir_name = "Rowan__hellaswag"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation", "test"]

    @staticmethod
    def custom_preprocessing(row):
        row["question"] = row["ctx"]
        row["options"] = "A. {} B. {} C. {} D. {}".format(
            row["endings"][0], row["endings"][1], row["endings"][2], row["endings"][3]
        )
        row["prompt"] = format_mcq(row["question"], row["options"])
        answer = row["label"]
        row["gold"] = chr(ord("A") + answer) if answer in [0, 1, 2, 3] else None


class Winogrande(Benchmark):

    """
    Huggingface card: https://huggingface.co/datasets/winogrande
    """

    def __init__(self, name="winogrande") -> None:
        super().__init__(name)
        self.hub_name = "winogrande"
        self.dir_name = "winogrande"
        self.path = os.path.join(ROOT_DIR, "benchmarks", "datasets", self.dir_name)
        self.splits = ["train", "validation"]
        self.subsets = ["winogrande_debiased"]

    @staticmethod
    def custom_preprocessing(row):
        row["question"] = row["sentence"]
        row["options"] = "A. {} B. {}".format(row["option1"], row["option2"])
        row["prompt"] = format_mcq(row["question"], row["options"])
        answer = row["answer"]
        row["gold"] = chr(ord("A") + answer) if answer in [1, 2] else None
        return row


def format_mcq(question, options):
    """
    Formats a multiple choice question with the given options.
    Uses the format recommended by: https://huggingface.co/blog/evaluating-mmlu-leaderboard

    'Question: What is the capital of France?

    Options:
    A. London
    B. Paris
    C. Berlin
    D. Rome'

    :param question: str, the question
    :param options: list of str, the options
    :return: str, the formatted question
    """
    if not question.endswith("?") and not question.endswith("."):
        question += "?"
    options_str = "\n".join([f"{chr(65+i)}. {options[i]}" for i in range(len(options))])
    prompt = "Question: " + question + "\n\nOptions:\n" + options_str
    return prompt


def aggregate_datasets(path, subsets, partition="train"):
    """
    Takes as input a Huggingface DatasetDict with subset name as key, and Dataset as value.
    Returns a pd.DataFrame with all subsets concatenated.

    :param subsets: list of str, the subsets of the data to download from the HuggingFace hub.
    :return: pd.DataFrame
    """
    dataframes = []
    for subset in subsets:
        subset_data = load_dataset(os.path.join(path, subset), split=partition)
        subset_df = pd.DataFrame(subset_data.map(lambda x: {"subset": subset, **x}))
        dataframes.append(subset_df)
    aggregate_df = pd.concat(dataframes, axis=0)
    aggregate = Dataset.from_pandas(aggregate_df)
    if "__index_level_0__" in aggregate.column_names:
        aggregate = aggregate.remove_columns("__index_level_0__")
    return aggregate


def model_setting(model_name, API_KEY):
    model = dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)
    return model


def hfmodel_setting(model_name):
    model = dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)
    return model


def together_setting(model_name, API_KEY):
    model = dspy.Together(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)
    return model


def answer_prompt(prompts, model):
    responses = []
    for prompt in tqdm(prompts, desc="Generating Responses", unit="prompt"):
        pred_response = model(prompt)
        generated_response = pred_response[0]
        responses.append(generated_response)
    return responses


def benchmark_preparation(benchmark_obj, dspy_module, shots):
    for partition in benchmark_obj.splits:
        benchmark_obj.load_data(partition=partition)
        benchmark_obj.preprocessing(partition=partition)
    if dspy_module is None and shots > 0:
        logging.info("Loading train data for few shot learning")
        benchmark_obj.load_data(partition="train")
        benchmark_obj.preprocessing(partition="train")
        logging.info(f"FEW SHOTS: {shots}")
        benchmark_obj.add_few_shot(shots=shots)


def evaluate_model(dspy_module, benchmark_instance, model):
    try:
        # Generate the training set
        trainset = dspy_module.store_correct_cot(
            benchmark_instance.train_data["input"]
            if "input" in benchmark_instance.train_data.column_names
            else benchmark_instance.train_data["question"],
            benchmark_instance.train_data["optionsKey"]
            if "optionsKey" in benchmark_instance.train_data.column_names
            else benchmark_instance.train_data["options"],
            benchmark_instance.train_data["gold"],
        )
        # Initialize MedpromptModule with trainset and shots
        # print(trainset)
        module = dspy_module(trainset=trainset, shots=5)
        predictions = []
        # Generating predictions
        for question, options in tqdm(
            zip(
                benchmark_instance.test_data["prompt"],
                benchmark_instance.test_data["optionsKey"]
                if "optionsKey" in benchmark_instance.test_data.column_names
                else benchmark_instance.test_data["options"],
            ),
            desc="Generating Responses",
            unit="prompt",
        ):
            response = module(question, options)
            predictions.append(response)
    except Exception as e:
        print(f"An error occurred while instantiating the module: {e}")
        module = DefaultModule()
        predictions = []
        # Generating predictions
        for question, options in tqdm(
            zip(
                benchmark_instance.test_data["prompt"],
                benchmark_instance.test_data["optionsKey"]
                if "optionsKey" in benchmark_instance.test_data.column_names
                else benchmark_instance.test_data["options"],
            ),
            desc="Generating Responses",
            unit="prompt",
        ):
            response = module(question, options)
            predictions.append(response.answer)
        # predictions = answer_prompt(benchmark_instance.test_data["prompt"][:10], model)
    print(predictions)
    evaluate_predictions(predictions, benchmark_instance.test_data["gold"])


def evaluate_predictions(pred, ref):
    correct = sum(1 for pred_letter, truth in zip(pred, ref) if pred_letter[0] == truth)
    total = len(ref)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")


def test(model, api_key, dspy_module, benchmark, shots):
    if model in ["gpt-3.5-turbo", "gpt-4-turbo-preview"]:
        model = model_setting(model, api_key)
    elif len(api_key) > 10:
        model = together_setting(model, api_key)
    else:
        model = hfmodel_setting(model)

    # Creating a benchmark instance, loading data and processing.
    benchmark_instance = benchmark_factory(benchmark)
    benchmark_preparation(benchmark_instance, dspy_module, shots)

    # Evaluating
    evaluate_model(dspy_module, benchmark_instance, model)
