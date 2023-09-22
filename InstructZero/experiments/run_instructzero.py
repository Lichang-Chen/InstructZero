import random
import torch
import numpy as np
import copy
from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re
from misc import get_test_conf, get_conf

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from instruction_coupled_kernel import *
import time

from misc import set_all_seed, TASKS, tkwargs, N_INIT, BATCH_SIZE, N_ITERATIONS
from args import parse_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

    
class LMForwardAPI:
    def __init__(self, model_name=None, eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None, 
                 HF_cache_dir=None, args=None):
        p = torch.ones(10)
        
        kwargs={
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        self.ops_model = model_name
        import pdb; pdb.set_trace()
        if self.ops_model in ["vicuna", "wizardlm", 'openchat']:
            self.model = AutoModelForCausalLM.from_pretrained(
                                HF_cache_dir, low_cpu_mem_usage=True, **kwargs
                            ).cuda()

            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=1024,
                                padding_side="left",
                                use_fast=False,
                            )
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if self.ops_model in ['wizardlm', 'vicuna', 'openchat']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        # Create the template for Vicuna and WizardLM
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        if self.ops_model == 'vicuna':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'wizardlm':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'alpaca':
            self.system_prompt= "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        else:
            NotImplementedError
            

        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['wizardlm', 'vicuna', 'openchat']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)
        elif random_proj == 'uniform':  
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)

        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        # Temporarily remove the API model "LLaMA-33B" and "Flan-T5 13B" 
        # if args.api_model in ['llama', 'flan-t5']:
        #     self.api_model = exec_evaluator(args.api_model, self.conf)
        # else:
        self.api_model = args.api_model

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
    
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        # create the input text with the system prompt  
        input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embed = torch.cat((prompt_embedding, input_embed), 1)

        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=128)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # postprocess instruction
        # instruction[0] = 'The instruction was to ' + instruction[0]
        # import pdb; pdb.set_trace()
        # start = instruction[0].find('The instruction was to')
        # end = instruction[0].find('Comment:')
        # if end == -1:
        #     instruction[0] = instruction[0][start:]
        # else:
        #     instruction[0] = instruction[0][start: end]

        # sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        # search_string = 'The instruction was to'
        # for sentence in sentences:
        #     if sentence.startswith(search_string):
        #         instruction[0] = sentence
        #         break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))
        
        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if self.api_model in ['chatgpt']: 
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            # We will fix the bugs for other api models. Stay tuned!
            # elif api_model in ['llama', 'flan-t5']: 
            #     dev_perf, instruction_score = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data,
            #                             self.conf['evaluation']).sorted()[1][0]            
            #     self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set
    
def run(args):
    task, HF_cache_dir=args.task, args.HF_cache_dir
    random_proj, intrinsic_dim, n_prompt_tokens= args.random_proj, args.intrinsic_dim, args.n_prompt_tokens

    assert args.task in TASKS, 'Task not found!'

    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]" # change the evaluation template
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to?"
    # prompt_gen_template = "[full_DEMO]\n\nWhat was the instruction for the task?"
    # prompt_gen_template = "[full_DEMO]\n\n Please generate appropriate instructions for the task."

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = get_conf(task, eval_data)

    # make the demo automatically
    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]

    
    model_forward_api = LMForwardAPI(model_name=args.model_name, eval_data=eval_data, init_prompt=init_prompt, 
                                    init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data, random_proj=random_proj, 
                                    intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, args=args)
    
        
    # start bayesian opt
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    X_return = [model_forward_api.eval(x) for x in X]
    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]
    
    
    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)
    print(f"Best initial point: {Y.max().item():.3f}")


    # standardization Y (no standardization for X)
    X_train = X
    y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2))

    # define matern kernel
    matern_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=X_train.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
    matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
    
    covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    
    
    
    for i in range(N_ITERATIONS):
        print(f"X_train shape {X_train.shape}")
        print(f"y_train shape {y_train.shape}")

        start_time = time.time()

        fit_gpytorch_model(gp_mll)#, options = {'maxiter':10})
        print(f"Fitting done in {time.time()-start_time}")
        start_time = time.time()
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        
        starting_idxs = torch.argsort(-1*y_train)[:BATCH_SIZE]
        starting_points = X_train[starting_idxs]


        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
            
        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        print(f"Time for CMA-ES {time.time() - start_time}")
        for idx in np.argsort(-1*np.array(best_vals)):
            X_next_point =  torch.from_numpy(best_points[idx]).float().unsqueeze(0)
            # Y_next_point = [model_forward_api.eval(X_next_point)]
            
            X_next_points_return = [model_forward_api.eval(X_next_point)]
            Y_next_point = [X[0] for X in X_next_points_return]
            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]
    

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])


        # standardization Y
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2))

        
        matern_kernel = MaternKernel(
                        nu=2.5,
                        ard_num_dims=X_train.shape[-1],
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    )
        
        matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )

        covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))

        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    
        print(f"Best value found till now: {torch.max(Y)}")

    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())

    # Evaluate on test data
    print('Evaluating on test data...')

    test_conf = get_test_conf(task, test_data)
    
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_conf)
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    return test_score
    # print(f'Test score on ChatGPT: {test_score}')


if __name__ == '__main__':
    args = parse_args()
    # evaluation budget
    print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")
    print(set_all_seed(args.seed))
    test_score = run(args=args)
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')


