import argparse
from datasets import DatasetDict, load_dataset
import dspy
import openai
import faiss
from tqdm import tqdm
import random
from dspy.teleprompt import KNNFewShot
from dspy.predict.knn import KNN
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate import Evaluate


def model_setting(model_name, API_KEY):

    model=dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)

def hfmodel_setting(model_name):

    model=dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)
# Formatting functions for different datsets.
#MedQA
def formatting_options(qoptions):
  return [' '.join(f"{option['key']} {option['value']}" for option in options) for options in qoptions]

def convert_format(input_options: list[dict]):
    result = []
    
    for dictionary in input_options:
        converted_list = []
        
        for key, value in sorted(dictionary.items()):
            converted_list.append({'key': key, 'value': value})
        
        result.append(converted_list)
    
    return result

# To be used for generating chain of thought which are to be stored.
class MultipleChoiceQA(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(desc="The set of options in the format : A option1 B option2 C option3 D option4 E option5 where A corresponds to option1, B to option2 and so on.")
    answer = dspy.OutputField(desc="A single-letter answer corresponding to the selected option.")

# To be used for answering the test question.
class MultipleChoiceQA1(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(desc="The set of options in the format : A option1 B option2 C option3 D option4 E option5 where A corresponds to option1, B to option2 and so on.")
    context = dspy.InputField(desc="may contain relevant facts")
    answer = dspy.OutputField(desc="A single-letter answer corresponding to the selected option.")


generate_answer = dspy.ChainOfThought(MultipleChoiceQA)
def store_correct_cot(questions: list[str], option_sets: list[str], answers: list[str]) -> list[str]:
    train_set = []
    for question, options, answer in tqdm(zip(questions, option_sets, answers),desc="Generating COTs", unit="cot"):
        pred_response = generate_answer(question=question, options=options)
        if pred_response.answer[0] == answer:
          example = dspy.Example(
            question=question,
            options=options,
            # Commented out for evaluate_medprompt
            # context=pred_response.rationale.split('.', 1)[1].strip(),
            context=pred_response.rationale,
            answer=answer
        ).with_inputs("question", "options")

          train_set.append(example)

    return train_set

class MultipleQABot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MultipleChoiceQA1)

    def forward(self, question, options):
        answer = self.generate_answer(question=question,options=options)

        return answer


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

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


def ask_questions(program):
    while True:
        q = input("Ask me a question (type 'exit' to end):\n")
        if q.lower() == 'exit':
            break

        o = input("Please provide the options.\n")
        pred = program(question=q, options=o)
        print(pred.answer)   

def main(args):

    if args.model == "gpt-3.5-turbo" or args.model == "gpt-4":
        model_setting(args.model, args.api_key)
    else:
        hfmodel_setting(args.model)
    
    #Loading the dataset
    dataset: DatasetDict = (load_dataset(args.dataset))

    #Processing datset
    train_subset = dataset["train"]
    train_questions = train_subset["question"]
    train_options = train_subset["options"]
    train_answers_id = train_subset["answer_idx"]

    #Formatting input data
    if args.dataset == "GBaker/MedQA-USMLE-4-options":
        formatted_train_options = formatting_options(convert_format(train_options))

    if args.dataset == "bigbio/med_qa":
        formatted_train_options = formatting_options(train_options)


    #Generating chain of thought
    trainset = store_correct_cot(train_questions, formatted_train_options, train_answers_id)

    #KNN Fewshot
    knn_teleprompter = KNNFewShot(KNN, args.shots, trainset)
    compiled_knn = knn_teleprompter.compile(MultipleQABot(), trainset=trainset)

    #Ensemble
    programs = [compiled_knn]
    ensembled_program = Ensemble(reduce_fn=dspy.majority).compile(programs)

    ask_questions(ensembled_program)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--dataset", type=str, default="GBaker/MedQA-USMLE-4-options")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of shots for knn fewshot.")

    args = parser.parse_args()
    main(args)