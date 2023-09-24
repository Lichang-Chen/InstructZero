import torch
import random
import numpy as np
from evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
import os

TASKS=[
    'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
    'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'translation_en-fr', 'word_in_context', 'auto_categorization', 'auto_debugging', 'ascii', 'cs_algorithms',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out', 'object_count'
]


SMOKE_TEST = os.environ.get("SMOKE_TEST")
## bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


N_INIT = 25
N_ITERATIONS = 5 if not SMOKE_TEST else 1
BATCH_SIZE = 25 if not SMOKE_TEST else 1


def get_test_conf(task, test_data):
    test_conf={
                'generation': {
                    'num_subsamples': 3,
                    'num_demos': 5,
                    'num_prompts_per_subsample': 0,
                    'model': {
                        'gpt_config': {
                            # 'model': 'text-ada-001'
                        }
                    }
                },
                'evaluation': {
                    'method': exec_accuracy_evaluator, # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
                    'num_samples': min(100, len(test_data[0])),
                    'task': task,
                    'model': {
                        "name": "GPT_forward",
                        'gpt_config': {
                            'model': 'GPT-3.5-turbo',
                        }
                    }
                }
            }
    return test_conf


def get_conf(task, eval_data):
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 20,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }
    return conf



def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"