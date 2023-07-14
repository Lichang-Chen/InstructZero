import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default="uniform",
        help="The initialization of the projection matrix A."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=10,
        help="The instrinsic dimension of the projection matrix"
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="/data/bobchen/vicuna-13b",
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Set the alpha if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Set the beta if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default='chatgpt',
        help="The black-box api model."    
    )
    args = parser.parse_args()
    return args