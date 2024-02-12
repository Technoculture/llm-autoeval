import dspy
from benchmark import benchmark_factory
import logging
from tqdm import tqdm

def model_setting(model_name, API_KEY):

    model=dspy.OpenAI(model=model_name, api_key=API_KEY)
    dspy.settings.configure(lm=model)
    return model

def hfmodel_setting(model_name):

    model=dspy.HFModel(model=model_name)
    dspy.settings.configure(lm=model)
    return model

def benchmark_preparation(benchmark_obj, dspy_module, shots):

    for partition in benchmark_obj.splits:
        benchmark_obj.load_data(partition=partition)
        benchmark_obj.preprocessing(partition=partition)
    if dspy_module == None and shots > 0:
            logging.info('Loading train data for few shot learning')
            benchmark_obj.load_data(partition='train')
            benchmark_obj.preprocessing(partition='train')
            logging.info(f'FEW SHOTS: {shots}')
            benchmark_obj.add_few_shot(
                    shots=shots)

def evaluate_model(dspy_module, benchmark_instance):

    # Generate the training set
    trainset = dspy_module.store_correct_cot(benchmark_instance.train_data["question"], benchmark_instance.train_data["optionsKey"], benchmark_instance.train_data["gold"])
    # Initialize MedpromptModule with trainset and shots
    medprompt_module = dspy_module(trainset=trainset, shots=5)

    predictions = []
    # Generating predictions
    for question, options in tqdm(zip(benchmark_instance.test_data["prompt"], benchmark_instance.test_data["optionsKey"]), desc="Generating Responses", unit="prompt"):
        response = medprompt_module(question, options)
        predictions.append(response)

    evaluate_predictions(predictions, benchmark_instance.test_data["gold"])

def evaluate_predictions(pred, ref):


    correct = sum(1 for pred_letter, truth in zip(pred, ref) if pred_letter[0] == truth)
    total = len(ref)
    accuracy = (correct / total)
    print(f"Accuracy: {accuracy:.2%}")

def test(model, api_key, dspy_module, benchmark, shots):

    if model == "gpt-3.5-turbo" or model == "gpt-4":
        model = model_setting(model, api_key)
    else:
        model = hfmodel_setting(model)

    # Creating a benchmark instance, loading data and processing.
    benchmark_instance = benchmark_factory(benchmark)
    benchmark_preparation(benchmark_instance, dspy_module, shots)

    # Evaluating
    evaluate_model(dspy_module, benchmark_instance)
