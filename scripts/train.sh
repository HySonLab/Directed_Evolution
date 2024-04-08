dataset=$1
devices=$2
batch_size=${3:-128}
ckpt_path=${4:-''}

data_file="/home/thanhtvt1/workspace/Latent-Based-Directed-Evolution/preprocessed_data/${dataset}/${dataset}.csv"
pretrained_encoder="facebook/esm2_t12_35M_UR50D"
dec_hidden_dim=1280
lr=0.0002
num_epochs=100
num_ckpts=3
precision="highest"

python train_decoder.py --data_file $data_file --dataset_name $dataset \
                        --pretrained_encoder $pretrained_encoder --dec_hidden_dim $dec_hidden_dim \
                        --batch_size $batch_size --devices $devices \
                        --lr $lr --num_epochs $num_epochs --num_ckpts $num_ckpts \
                        --precision $precision #--ckpt_path=$ckpt_path