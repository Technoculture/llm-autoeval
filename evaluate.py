import argparse
import openai
import dspy
from benchmark import benchmark_factory
import torch

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
    answer = dspy.OutputField(desc="A single-letter answer corresponding to the selected option.")

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
    hfmodel_setting(args.model)

    #Creating a benchmark instance, loading data and processing.
    partition = "validation"
    benchmark_instance = benchmark_factory("medmcqa")
    benchmark_instance.load_data(partition=partition)
    benchmark_instance.preprocessing(partition=partition)
    questions = benchmark_instance.test_data["question"][:20]
    options = benchmark_instance.test_data["options"][:20]
    
    #Evaluating
    evaluate_model(benchmark_instance, questions, options)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Benchmark name.", default="arc")
    args = parser.parse_args()
    main(args)
