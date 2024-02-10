import argparse
import openai
import dspy
from benchmark import benchmark_factory
from medprompt import store_correct_cot, MultipleQABot, Ensemble
import logging
import re
import random
from tqdm import tqdm
from dspy.teleprompt import KNNFewShot
from dspy.predict.knn import KNN

def model_setting(model_name, API_KEY):

    model=dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)
    return model

def hfmodel_setting(model_name):

    model=dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)
    return model


def answer_prompt(prompts, model):
    responses = []
    for prompt in tqdm(prompts, desc="Generating Responses", unit="prompt"):
        pred_response = model(prompt)
        generated_response = pred_response[0]
        responses.append(generated_response)
    return responses

def medprompt(test_set, train_set, model):
    responses = []
    trainset = store_correct_cot(train_set["question"][:50], train_set["optionsKey"][:50], train_set["gold"][:50])
    for prompt, options in tqdm(zip(test_set["prompt"][:50], test_set["optionsKey"][:50]), desc="Generating Responses", unit="prompt"):

        #KNN Fewshot
        knn_teleprompter = KNNFewShot(KNN, args.shots, trainset)
        compiled_knn = knn_teleprompter.compile(MultipleQABot(), trainset=trainset)

        #Ensemble
        programs = [compiled_knn]
        ensembled_program = Ensemble(reduce_fn=dspy.majority).compile(programs)
        pred_response = ensembled_program(question=prompt, options=options)
        generated_response = pred_response.answer
        responses.append(generated_response)
    return responses

def benchmark_preparation(benchmark_obj, args):

    for partition in benchmark_obj.splits:
        benchmark_obj.load_data(partition=partition)
        benchmark_obj.preprocessing(partition=partition)
    if args.shots > 0:
            logging.info('Loading train data for few shot learning')
            benchmark_obj.load_data(partition='train')
            benchmark_obj.preprocessing(partition='train')
            logging.info(f'FEW SHOTS: {args.shots}')
            benchmark_obj.add_few_shot(
                    shots=args.shots)

def evaluate_model(args, benchmark_instance, model):

    if args.dspy_module =="medprompt":
        predictions = medprompt(benchmark_instance.test_data, benchmark_instance.train_data,  model)
    else:
        predictions = answer_prompt(benchmark_instance.test_data["prompt"][:50], model)
    print(predictions)
    evaluate_predictions(predictions, benchmark_instance.test_data["gold"][:50])

def evaluate_predictions(pred, ref):


    correct = sum(1 for pred_letter, truth in zip(pred, ref) if pred_letter[0] == truth)
    total = len(ref)
    accuracy = (correct / total)
    print(f"Accuracy: {accuracy:.2%}")

def main(args):

    if args.model == "gpt-3.5-turbo" or args.model == "gpt-4":
        model = model_setting(args.model, args.api_key)
    else:
        model = hfmodel_setting(args.model)

    # Creating a benchmark instance, loading data and processing.
    benchmark_instance = benchmark_factory(args.benchmark)
    benchmark_preparation(benchmark_instance, args)

    # Evaluating
    evaluate_model(args, benchmark_instance, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Choose one of the following benchmark: [medmcqa, medicationqa, mmlu_medical, mmlu_general, arc, hellaswag, winogrande, blurb, truthfulqa, gsm8k].", default="arc")
    parser.add_argument("--shots", type=int, help = "Number of few shots.", default=5)
    parser.add_argument("--dspy_module", type = str, help = "Name of dspy module.", default="medprompt")
    args = parser.parse_args()
    main(args)
