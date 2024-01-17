import argparse
import openai
import dspy
from benchmark import benchmark_factory
import torch
import logging

def model_setting(model_name, API_KEY):

    model=dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)

def hfmodel_setting(model_name):

    model=dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)

class MultipleChoiceQA(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(desc="The set of options in the format : A option1 B option2 and so on where A corresponds to option1, B to option2 and so on.")
    answer = dspy.OutputField(desc="First entry being a single-letter answer corresponding to the selected option only.")

class BaseQA(dspy.Signature):

    question = dspy.InputField(desc="Question.")
    options = dspy.InputField(desc="Options.")
    answer = dspy.OutputField(desc="Answer.")

class MultipleQABot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MultipleChoiceQA)

    def forward(self, question, options):
        answer = self.generate_answer(question=question,options=options)

        return answer

generate_answer = MultipleQABot()
def generate_responses(questions, option_sets):
    responses = []
    for question, options_list in zip(questions, option_sets):
        pred_response = generate_answer(question=question, options=options_list)
        generated_response = pred_response.answer
        responses.append(generated_response)
    return responses

def benchmark_preparation(benchmark_obj, args):

    partition = "test" if "test" in benchmark_obj.splits else "validation"
    benchmark_obj.load_data(partition=partition)
    benchmark_obj.preprocessing(partition=partition)
    if args.shots > 0:
            logging.info('Loading train data for few shot learning')
            benchmark_obj.load_data(partition='train')
            benchmark_obj.preprocessing(partition='train')
            for seed in [1234, 432, 32]:
                logging.info(f'Start seed {seed})')
                logging.info(f'FEW SHOTS: {args.shots}')
                benchmark_obj.add_few_shot(
                    shots=args.shots,
                    seed=seed)

def evaluate_model(benchmark_instance, questions, options):

    # benchmark_instance.load_data(partition=partition)
    # benchmark_instance.preprocessing(partition=partition)
    # questions = benchmark_instance.test_data["question"][:5]
    # options = benchmark_instance.test_data["options"][:5]
    predictions = generate_responses(questions, options)
    print(predictions)
    evaluate_predictions(predictions, benchmark_instance.test_data["answerKey"][:20])

def evaluate_predictions(pred, ref):

    correct = sum(1 for pred, truth in zip(pred, ref) if pred[0] == truth)
    total = len(ref)
    accuracy = (correct / total)
    print(f"Accuracy: {accuracy:.2%}")

def main(args):

    if args.model == "gpt-3.5-turbo" or args.model == "gpt-4":
        model_setting(args.model, args.api_key)
    else:
        hfmodel_setting(args.model)

    #Creating a benchmark instance, loading data and processing.
    benchmark_instance = benchmark_factory(args.benchmark)
    benchmark_preparation(benchmark_instance, args)

    #Defining test set.
    questions = benchmark_instance.test_data["question"][:20]
    options = benchmark_instance.test_data["optionsKey"][:20]
    
    #Evaluating
    evaluate_model(benchmark_instance, questions, options)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Choose one of the following benchmark: [medmcqa, medicationqa, mmlu_medical, mmlu_general, arc, hellaswag, winogrande, blurb, truthfulqa, gsm8k].", default="arc")
    parser.add_argument("--shots", type=int, help = "Number of few shots.", default=25)
    args = parser.parse_args()
    main(args)
