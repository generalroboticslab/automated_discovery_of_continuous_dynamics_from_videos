dataset=$1
trial=$2
seed=$3
gpu=$4

echo "=========================================================================================="
echo "============== Running $trial seed $seed on: $dataset (gpu id: $gpu) =============="
echo "=========================================================================================="

screen -S $dataset-$trial-$seed -dm bash -c "bash scripts/$trial.sh $dataset $seed $gpu";
