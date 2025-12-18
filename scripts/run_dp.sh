
# Data Generation + Model Training
gpu_ids=${1:-0,1,2}
task=${2:-mnli}
eps=${3:-6}

learning_rate=0.5
max_grad_norm=2
dist="non_iid"
alpha=0.5
n_clients=6
frac=0.5
n_iter=10

# Model Training
n_rnds=100
n_eps=1
model="ffa"
backbone="roberta-large"
peft="lora"
lora_rank=8
lora_alpha=8
batch_size=64
accumulation_steps=2
scheduler_type="constant"
recalculate_svd_period=1

# Misc
report_period=20

# python -m dataset.generator --dataset $task --n-clients $n_clients --alpha $alpha --seed $seed # You can comment this out if you already have the dataset
trial="K=${n_clients}_dp_${model}-svd_${task}_${learning_rate}_${eps}_C_${max_grad_norm}"
for seed in {42..46}
do
	python3 main.py --gpu $gpu_ids\
					--n-workers $n_clients\
					--n-clients $n_clients\
					--frac $frac\
					--lr $learning_rate\
					--lora-alpha $lora_alpha\
					--model $model\
					--task $task\
					--dist $dist\
					--trial $trial\
					--peft $peft\
					--n-rnds $n_rnds\
					--n-eps $n_eps\
					--alpha $alpha\
					--seed $seed\
					--lora-rank $lora_rank\
					--batch_size $batch_size\
					--backbone $backbone\
					--scheduler-type $scheduler_type\
					--accumulation-steps $accumulation_steps\
					--report-period $report_period\
					--recalculate-svd-period $recalculate_svd_period\
					--dropout 0.05 \
					--n-iter $n_iter\
					--dp\
					--delta 1e-5\
					--eps $eps\
					--max-grad-norm $max_grad_norm
done