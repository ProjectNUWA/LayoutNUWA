MODEL_PATH=/path/to/trained_model
DATA=/path/to/testing_file
OUTPUT_PATH=/path/to/output_dir

mkdir -p ${OUTPUT_PATH}

python generate.py --file_path ${DATA} --base_model ${MODEL_PATH} --output_dir ${OUTPUT_PATH}

