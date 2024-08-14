DATA_NAME=$1
USED_GPU=$2
PORT=$3

python main.py --cuda --is_offline --gpuid ${USED_GPU} --config ${DATA_NAME} -p ${PORT} -l