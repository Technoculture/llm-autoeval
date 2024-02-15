import argparse
from benchmark import test
from medprompt import MedpromptModule


def main(args):
    test(
        model=args.model,
        api_key=args.api_key,
        dspy_module=args.dspy_module,
        benchmark=args.benchmark,
        shots=args.shots,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="google/flan-t5-base", help="Model to be used."
    )
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Choose one of the following benchmark: [medmcqa, medicationqa, mmlu_medical, mmlu_general, arc, hellaswag, winogrande, blurb, truthfulqa, gsm8k].",
        default="arc",
    )
    parser.add_argument("--shots", type=int, help="Number of few shots.", default=5)
    parser.add_argument(
        "--dspy_module", type=MedpromptModule, help="Name of dspy module.", default=MedpromptModule
    )
    args = parser.parse_args()
    main(args)
