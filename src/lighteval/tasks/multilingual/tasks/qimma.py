"""
name:
Qimma Leaderboard task evals

dataset:
qimma/*

abstract:
Collection of benchmarks for Arabic language.

languages:
arabic

tags:
knowledge, multilingual, multiple-choice

paper: TBD
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.dynamic_metrics import NormalizedMultiChoiceScoreMetric, NormalizedMultiChoiceProbMetric
import numpy as np

# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on

def qimma_pfn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line['prompt'],
        choices=LETTER_INDICES_AR[:len(line['choices'])],
        gold_index=line['index'],
        instruction=None,
    )

def multi_choice_scorer(gold_index, log_probs):
    ## sort indexs based on log_prob in ascending manner
    sort_args = np.argsort(log_probs)
    i = len(log_probs) - len(gold_index)
    # args with highest scores (represent top choices)
    best_args = sort_args[i:]
    best_args.sort()

    return int(np.array_equal(sorted(gold_index), best_args))

class CustomQimmaNativeTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
        hf_repo,
        multi_select = False
    ):
        metrics = [Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})]
        if multi_select:
            score_based_met = NormalizedMultiChoiceScoreMetric(
                normalization = LogProbCharNorm(),
                score_function= multi_choice_scorer

            )

            prob_based_met = NormalizedMultiChoiceProbMetric(
                normalization = LogProbCharNorm(),
                aggregation_function = np.sum

            )

            metrics = [score_based_met, prob_based_met]
            
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            hf_repo=hf_repo,
            prompt_function=qimma_pfn,
            metrics=metrics,
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )

def construct_tasks_from_subsets(hf_repo, benchmark, subsets):
    return [CustomQimmaNativeTask(name=f"qimma-{benchmark}:{subset}", hf_subset=subset, hf_repo=hf_repo) for subset in subsets]


QIMMA_BENCHMARKS = ['AraTrust', 'MizanQA', 'NativeQA-RDP', 'NativeQA', 'PALMX-2025']
QIMMA_TASKS = [CustomQimmaNativeTask(name=f"qimma-{benchmark}", hf_subset="default", hf_repo=f"qimma/MCQ_{benchmark}") for benchmark in QIMMA_BENCHMARKS]

mizan_task = CustomQimmaNativeTask(name=f"qimma-mizan", hf_subset="default", hf_repo=f"amztheory/MizanQA-v0", multi_select=True)
QIMMA_TASKS.append(mizan_task)


AraDiCE_Subsets = ['Egypt', 'Jordan', 'Lebanon', 'Palestine', 'Qatar', 'Syria']
AraDiCE_Tasks = construct_tasks_from_subsets("qimma/MCQ_AraDiCE-Culture","AraDiCE-Culture", AraDiCE_Subsets)

Arabculture_Subsets = ['Algeria', 'Egypt', 'Jordan', 'KSA', 'Lebanon', 'Libya', 'Morocco', 'Palestine', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen']
Arabculture_Tasks = construct_tasks_from_subsets("qimma/MCQ_ArabCulture","ArabCulture", Arabculture_Subsets)

MedArabiQ_Subsets = ['fib_with_choices', 'mcq_bias', 'mcq_knowledge']
MedArabiQ_Tasks = construct_tasks_from_subsets("qimma/MCQ_MedArabiQ","MedArabiQ", MedArabiQ_Subsets)

SyntheticQA_Subset = ['Biology', 'Chemistry', 'General_Science', 'Math', 'Physics']
SyntheticQA_Tasks = construct_tasks_from_subsets("qimma/MCQ_SyntheticQA", "SyntheticQA", SyntheticQA_Subset)

ArabicMMLU_subsets = ['Arabic Language (Middle School)', 'Civics (High School)', 'Social Science (Middle School)', 'Economics (High School)', 'History (High School)', 'Political Science (University)', 'Geography (High School)', 'Islamic Studies (High School)', 'Arabic Language (Primary School)', 'Natural Science (Primary School)', 'Philosophy (High School)', 'General Knowledge', 'Arabic Language (High School)', 'Economics (University)', 'Islamic Studies (Primary School)', 'Geography (Middle School)', 'Islamic Studies', 'Biology (High School)', 'Natural Science (Middle School)', 'Islamic Studies (Middle School)', 'Math (Primary School)', 'Computer Science (Primary School)', 'Computer Science (High School)', 'Social Science (Primary School)', 'Arabic Language (Grammar)', 'Physics (High School)', 'History (Primary School)', 'Driving Test', 'Civics (Middle School)', 'History (Middle School)', 'General Knowledge (Middle School)', 'General Knowledge (Primary School)', 'Geography (Primary School)', 'Law (Professional)', 'Computer Science (University)', 'Accounting (University)', 'Economics (Middle School)', 'Management (University)', 'Computer Science (Middle School)', 'Arabic Language (General)']
ArabicMMLU_subsets = construct_tasks_from_subsets("qimma/MCQ_ArabicMMLU","ArabicMMLU", ArabicMMLU_subsets)

QIMMA_TASKS = (
    QIMMA_TASKS
    + AraDiCE_Tasks
    + Arabculture_Tasks
    + MedArabiQ_Tasks
    + SyntheticQA_Tasks
    + ArabicMMLU_subsets
)


TASKS_TABLE = (
    QIMMA_TASKS
)