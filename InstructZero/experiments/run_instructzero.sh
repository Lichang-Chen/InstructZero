export CUDA_VISIBLE_DEVICES=0
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10

datasets=(antonyms cause_and_effect common_concept diff)

for i in ${datasets[@]}; do
    echo $i
    python experiments/run_instructzero.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $sft \
    --intrinsic_dim $INTRINSIC_DIM
done