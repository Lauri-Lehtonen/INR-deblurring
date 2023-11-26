# sh-file used to run deblur program for wanted range within dataset using the wanted NN_model

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <python_file> <operation> <model> <dataset> <nn_model> <start_id> <end_id>"
    exit 1
fi

# parameters
python_file=$1
operation=$2
model=$3
dataset=$4
nn_model=$5
start_id=$6
end_id=$7

# Loop through the desired range of --id values
for ((id=start_id; id<=end_id; id++)); do
    echo "################################"
    echo "Running with --id $id"
    echo "################################"
    python3 "$python_file" "$operation" "--model" "$model" "--dataset" "$dataset" "--nn_model" "$nn_model" "--id" "$id"
done


echo "summary for dataset $dataset"
python3 summary.py "$operation" "--model" "$model" "--dataset" "$dataset"
