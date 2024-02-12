import argparse
from test import test
from med import MedpromptModule


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--benchmark", type=str, help = "Choose one of the following benchmark: [medmcqa, medicationqa, mmlu_medical, mmlu_general, arc, hellaswag, winogrande, blurb, truthfulqa, gsm8k].", default="arc")
    parser.add_argument("--shots", type=int, help = "Number of few shots.", default=5)
    parser.add_argument("--dspy_module", type = str, help = "Name of dspy module.", default="medprompt")
    args = parser.parse_args()

    results = test(
    model="google/flan-t5-base",
    api_key = "",
    dspy_module=MedpromptModule,
    benchmark="arc",
    shots=5,
    )

    print(results)

