export CUDA_VISIBLE_DEVICES=0

datasets=(antonyms cause_and_effect common_concept diff)

for i in ${datasets[@]}; do
    echo $i
    torchrun --nproc_per_node 1 --master_port 13452 experiments/run_instructzero.py --task $i
done