import argparse
from benchmark import benchmark_factory

def evaluate_model(model, benchmark_instance, partition='test'):

    benchmark_instance.load_data(partition=partition)
    benchmark_instance.preprocessing(partition=partition)
    input_data = benchmark_instance.test_data.map()
    predictions = model.predict(input_data[""])
    evaluate_predictions(predictions, benchmark_instance.test_data)

def evaluate_predictions(pred, ref):

    correct = sum(1 for pred, truth in zip(pred, ref) if pred == truth)
    total = len(ref)
    accuracy = (correct / total)*100
    print(f"Accuracy: {accuracy:.2%}")

def main(args):

    benchmark = benchmark_factory(args.benchmark)
    evaluate_model(args.model, benchmark)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Benchmark name.")
    args = parser.parse_args()
    main(args)