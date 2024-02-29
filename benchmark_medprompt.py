import argparse
from benchmark import test
from medprompt import MedpromptModule
from pydantic import BaseModel, Field, SkipValidation
from enum import Enum


class TestConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model: str = Field(
        default="google/flan-t5-base", description="The model to be used."
    )
    api_key: str = Field(default="API_KEY", description="Your API key.")
    benchmark: str = Field(default="arc", description="The benchmark option to choose.")
    shots: int = Field(default=5, description="Number of shots.")
    dspy_module: SkipValidation[MedpromptModule] = Field(
        default=MedpromptModule, description="The dspy module class to be used."
    )  # type: ignore


class BenchmarkOptions(str, Enum):
    medmcqa = "medmcqa"
    medicationqa = "medicationqa"
    mmlu_medical = "mmlu_medical"
    mmlu_general = "mmlu_general"
    arc = "arc"
    hellaswag = "hellaswag"
    winogrande = "winogrande"
    blurb = "blurb"
    truthfulqa = "truthfulqa"
    gsm8k = "gsm8k"
    openllm = "openllm"
    nous = "nous"
    medical = "medical"
    medical_openllm = "medical-openllm"
    mixqa = 'mixqa'


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
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY", default="api_key")
    parser.add_argument(
        "--benchmark",
        type=BenchmarkOptions,  # Use the BenchmarkOptions enum here
        choices=list(BenchmarkOptions),
        help="Choose one of the following benchmarks.",
        default="arc",
    )
    parser.add_argument("--shots", type=int, help="Number of few shots.", default=5)
    parser.add_argument(
        "--dspy_module",
        help="Name of dspy module.",
        default=MedpromptModule,
    )

    args = TestConfig(**vars(parser.parse_args()))
    results = main(args)
