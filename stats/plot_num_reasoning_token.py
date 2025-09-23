import argparse
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # LaTeX's default
    "font.size": 12,
    "text.latex.preamble": r"\usepackage{times}"  # Or any other package you use
})

MODELS = [
  "llama-3.1-8b",
  "llama-3.1-70b",
  "gpt-4.1-mini",
  "gpt-4.1",
  "QwQ-32B",
  "deepseek-reasoner",
  # "deepseek-chat",
  # "deepseek-R1-8b",
  # "deepseek-R1-32b",
  # "deepseek-R1-70b",
  # "o3-mini",
  # "o4-mini",
]

GREEN = (56 / 255, 118 / 255, 29 / 255)
# GREEN = (107 / 255, 174 / 255, 214 / 255)
RED = (200 / 255, 55 / 255, 40 / 255)

def main(args):
  fig, ax = plt.subplots(figsize=(5, 2.15))

  reasoning_token_results = []

  for (model_id, model) in enumerate(MODELS):
    base_filename = f"eval/{model}_prompt_base_report.json"
    base_results = json.load(open(base_filename))

    corrects = []
    incorrects = []
    for (case_name, case) in base_results["case_details"].items():
      for turn in case["turns"]:
        n = turn["n_reasoning_tokens"]
        if n == 1: continue
        if turn["is_correct"]:
          corrects.append(n)
        else:
          incorrects.append(n)

    vplot_correct = ax.violinplot(corrects, [model_id], widths=0.7,
                     showmedians=True, showextrema=True,
                     bw_method=0.5, side='low')

    vplot_incorrect = ax.violinplot(incorrects, [model_id], widths=0.7,
                        showmedians=True, showextrema=True,
                        bw_method=0.5, side='high')
    
    for body in vplot_correct['bodies']:
      body.set_facecolor(GREEN)
      body.set_edgecolor("black")
      body.set_alpha(0.5)

    for body in vplot_incorrect['bodies']:
      body.set_facecolor(RED)
      body.set_edgecolor("black")
      body.set_alpha(0.5)

    for key in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
      for vplot in [vplot_correct, vplot_incorrect]:
        vplot[key].set_linewidth(1.0)
        vplot[key].set_color("black")
        vplot[key].set_alpha(0.5)

  ax.set_yscale('log')
  ax.set_ylabel('\\# Reasoning Tokens')
  custom_ticks = [10, 50, 100, 500, 1000, 5000]
  ax.set_yticks(custom_ticks)
  ax.set_yticklabels([str(tick) for tick in custom_ticks])
  ax.yaxis.grid(True, which='both', linestyle='-.', alpha=0.7)

  ax.set_xticks([model_id for (model_id, _) in enumerate(MODELS)])
  ax.set_xticklabels(["L3.1-8B", "L3.1-70B", "G4.1-M", "G4.1", "Q-32B", "DS-R1"])
  ax.xaxis.grid(True, which='both', linestyle='-', alpha=0.7)

  legend_elements = [
    Patch(facecolor=GREEN, alpha=0.5, edgecolor='black', label='Correct Answers'),
    Patch(facecolor=RED, alpha=0.5, edgecolor='black', label='Incorrect Answers')
  ]
  ax.legend(handles=legend_elements, loc='upper left')
  ax.set_ylim(bottom=25)

  plt.savefig('stats/num_reasoning_tokens_violin.pdf', bbox_inches='tight')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args)
