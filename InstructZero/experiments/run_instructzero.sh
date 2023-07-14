export CUDA_VISIBLE_DEVICES=3
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
model_dir='WizardLM/WizardLM-13B-V1.1'

datasets=(antonyms cause_and_effect common_concept diff)

for i in ${datasets[@]}; do
    echo $i
    python experiments/run_instructzero.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir}
done