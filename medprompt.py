import dspy
from tqdm import tqdm
import random
from dspy.teleprompt import KNNFewShot
from dspy.predict.knn import KNN
from dspy.teleprompt.teleprompt import Teleprompter
from datasets import Dataset
import os

hf_token = os.environ.get("HF_TOKEN")


class MultipleChoiceQA(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(
        desc="The set of options in the format : A option1 B option2 C option3 D option4 E option5 where A corresponds to option1, B to option2 and so on."
    )
    answer = dspy.OutputField(
        desc="A single-letter answer corresponding to the selected option."
    )


# To be used for answering the test question.
class MultipleChoiceQA1(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(
        desc="The set of options in the format : A option1 B option2 C option3 D option4 E option5 where A corresponds to option1, B to option2 and so on."
    )
    context = dspy.InputField(desc="may contain relevant facts")
    answer = dspy.OutputField(
        desc="A single-letter answer corresponding to the selected option."
    )


def store_correct_cot(
    cls, questions: list[str], option_sets: list[str], answers: list[str]
) -> list[str]:
    train_set = []
    rows = []
    generate_answer = dspy.ChainOfThought(MultipleChoiceQA)
    for question, options, answer in tqdm(
        zip(questions, option_sets, answers), desc="Generating COTs", unit="cot"
    ):
        pred_response = generate_answer(question=question, options=options)
        if pred_response.answer[0] == answer:
            example = dspy.Example(
                question=question,
                options=options,
                # Commented out for evaluate_medprompt
                # context=pred_response.rationale.split('.', 1)[1].strip(),
                context=pred_response.rationale,
                answer=answer,
            ).with_inputs("question", "options")

            row = {
                "question": question,
                "options": options,
                "context": pred_response.rationale,
                "answer": answer,
            }

            train_set.append(example)
            rows.append(row)

    # with open('mixqa_correct_cot_trainset.json', 'w') as json_file:
    #     json.dump([vars(example) for example in train_set], json_file)

    # hf_dataset = Dataset.from_dict({'train':rows})
    hf_dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "options": [row["options"] for row in rows],
            "context": [row["context"] for row in rows],
            "answer": [row["answer"] for row in rows],
        }
    )
    print(hf_dataset)
    hf_dataset.push_to_hub("dkshjn/mixqa_cot_2", token=hf_token)

    return train_set


class MultipleQABot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MultipleChoiceQA1)

    def forward(self, question, options):
        answer = self.generate_answer(question=question, options=options)
        dspy.Suggest(
            len(answer) < 5, "Answer should be either one character or a short one."
        )

        return answer


class DefaultModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MultipleChoiceQA)

    def forward(self, question, options):
        answer = self.generate_answer(question=question, options=options)
        dspy.Suggest(
            len(answer) < 5, "Answer should be either one character or a short one."
        )

        return answer


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert (
            deterministic is False
        ), "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

        return EnsembledProgram()


class MedpromptModule(dspy.Module):
    store_correct_cot = classmethod(store_correct_cot)

    def __init__(self, trainset, shots):
        super().__init__()
        self.trainset = trainset
        self.shots = shots
        pass

    def forward(self, question, options):
        # KNN Fewshot
        knn_teleprompter = KNNFewShot(KNN, self.shots, self.trainset)
        compiled_knn = knn_teleprompter.compile(MultipleQABot(), trainset=self.trainset)

        # Ensemble
        # programs = [compiled_knn]
        # ensembled_program = Ensemble(reduce_fn=dspy.majority).compile(programs)
        pred_response = compiled_knn(question=question, options=options)
        generated_response = pred_response.answer
        return generated_response
