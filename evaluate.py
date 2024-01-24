import argparse
import openai
import dspy
from benchmark import benchmark_factory
import logging
import re
from tqdm import tqdm

def model_setting(model_name, API_KEY):

    model=dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)
    return model

def hfmodel_setting(model_name):

    model=dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)
    return model

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

def answer_prompt(prompts, model):
    responses = []
    for prompt in tqdm(prompts, desc="Generating Responses", unit="prompt"):
        pred_response = model(prompt)
        generated_response = pred_response[0]
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
            logging.info(f'FEW SHOTS: {args.shots}')
            benchmark_obj.add_few_shot(
                    shots=args.shots)
# for arc (phi-2)           
def format_answer(predictions):
    pred = []
    for prediction in tqdm(predictions, desc="Formatting Answers", unit="prediction"):
        matches = re.findall(r'The answer is: (\w)', prediction)
        if len(matches) >= 26:
            pred.append(matches[25])
        else:
            pred.append(None)
    return pred

# for truthfulqa (phi-2) 
def format_answer(predictions):
    pred = []
    for prediction in tqdm(predictions, desc="Formatting Answers", unit="prediction"):
        matches_option = re.findall(r'The correct option is (\w)', prediction)
        if matches_option:
            pred.append(matches_option[0])
        else:
            matches_answer = re.findall(r'The correct answer is (\w)', prediction)
            if matches_answer:
                pred.append(matches_answer[0])
            else:
                pred.append("None")
    
    return pred

def evaluate_model(benchmark_instance, model):

    predictions = answer_prompt(benchmark_instance.test_data["prompt"], model)
    print(predictions)
    if args.model == "microsoft./phi-2":
        predictions = format_answer(predictions)

    print(predictions)
    evaluate_predictions(predictions, benchmark_instance.test_data["gold"])

def evaluate_predictions(pred, ref):


    correct = sum(1 for pred_letter, truth in zip(pred, ref) if pred_letter[0] == truth)
    total = len(ref)
    accuracy = (correct / total)
    print(f"Accuracy: {accuracy:.2%}")

def main(args):

    # if args.model == "gpt-3.5-turbo" or args.model == "gpt-4":
    #     model = model_setting(args.model, args.api_key)
    # else:
    #     model = hfmodel_setting(args.model)

    #Creating a benchmark instance, loading data and processing.
    benchmark_instance = benchmark_factory(args.benchmark)
    benchmark_preparation(benchmark_instance, args)
    print(benchmark_instance.test_data["prompt"])
    #print(benchmark_instance.test_data["gold"])
    #Evaluating
    # evaluate_model(benchmark_instance, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Choose one of the following benchmark: [medmcqa, medicationqa, mmlu_medical, mmlu_general, arc, hellaswag, winogrande, blurb, truthfulqa, gsm8k].", default="arc")
    parser.add_argument("--shots", type=int, help = "Number of few shots.", default=0)
    args = parser.parse_args()
    main(args)
