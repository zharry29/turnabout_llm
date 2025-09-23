import argparse
import json
import math
from collections import defaultdict

MODELS = [
  "deepseek-reasoner",
  "gpt-4.1",
  "gpt-4.1-mini",
  "llama-3.1-8b",
  "llama-3.1-70b",
]

def main(args):
  acc_results = defaultdict(lambda: dict())

  for model in MODELS:
    base_filename = f"eval/{model}_prompt_base_report.json"
    base_result = json.load(open(base_filename))
    acc_results[model]["base_accuracy"] = base_result["overall_accuracy"]

    context_full_filename = f"eval/{model}_prompt_base_context_full_report.json"
    context_full_result = json.load(open(context_full_filename))
    acc_results[model]["full_context_accuracy"] = context_full_result["overall_accuracy"]

  lines = ["model,base_accuracy,full_context_accuracy,diff\n"]
  for (model, result) in acc_results.items():
    diff = result['full_context_accuracy'] - result['base_accuracy']
    lines.append(f"{model},{result['base_accuracy']},{result['full_context_accuracy']},{diff}\n")
  with open("stats/acc_vs_full_context.csv", "w") as f:
    f.writelines(lines)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args)
