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

paper: 
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


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

class CustomQimmaNativeTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
        hf_repo,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            hf_repo=hf_repo,
            prompt_function=qimma_pfn,
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )

QIMMA_BENCHMARKS = ['AraTrust', 'MizanQA', 'NativeQA-RDP', 'NativeQA', 'PALMX-2025']
QIMMA_TASKS = [CustomQimmaNativeTask(name=f"qimma-{benchmark}", hf_subset="default", hf_repo=f"qimma/MCQ_{benchmark}") for benchmark in QIMMA_BENCHMARKS]

AraDiCE_Subsets = ['Egypt', 'Jordan', 'Lebanon', 'Palestine', 'Qatar', 'Syria']
AraDiCE_Tasks = [CustomQimmaNativeTask(name=f"qimma-AraDiCE-Culture:{subset}", hf_subset=subset, hf_repo=f"qimma/MCQ_AraDiCE-Culture") for subset in AraDiCE_Subsets]
Arabculture_Subsets = ['Algeria', 'Egypt', 'Jordan', 'KSA', 'Lebanon', 'Libya', 'Morocco', 'Palestine', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen']
Arabculture_Tasks = [CustomQimmaNativeTask(name=f"qimma-ArabCulture:{subset}", hf_subset=subset, hf_repo=f"qimma/MCQ_ArabCulture") for subset in Arabculture_Subsets]
MedArabiQ_Subsets = ['fib_with_choices', 'mcq_bias', 'mcq_knowledge']
MedArabiQ_Tasks = [CustomQimmaNativeTask(name=f"qimma-MedArabiQ:{subset}", hf_subset=subset, hf_repo=f"qimma/MCQ_MedArabiQ") for subset in MedArabiQ_Subsets]
SyntheticQA_Subset = ['Biology', 'Chemistry', 'General_Science', 'Math', 'Physics']
SyntheticQA_Tasks = [CustomQimmaNativeTask(name=f"qimma-SyntheticQA:{subset}", hf_subset=subset, hf_repo=f"qimma/MCQ_SyntheticQA") for subset in SyntheticQA_Subset]

QIMMA_TASKS = QIMMA_TASKS + AraDiCE_Tasks + Arabculture_Tasks + MedArabiQ_Tasks + SyntheticQA_Tasks


TASK_TABLE = [
    QIMMA_TASKS,
]