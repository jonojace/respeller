cd /home/s1785140/respeller

# Experiment 1/8: CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=True
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=True \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384 \
    --freeze-embedding-table \
    --cross-entropy-loss \
    --pretrained-embedding-table  \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.1 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 2/8: CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=False
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=False \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384 \
    --freeze-embedding-table \
    --cross-entropy-loss   \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.1 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 3/8: CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=True
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=True \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384  \
    --cross-entropy-loss \
    --pretrained-embedding-table  \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.1 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 4/8: CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=False
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.1::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=False \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384  \
    --cross-entropy-loss   \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.1 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 5/8: CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=True
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=True \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384 \
    --freeze-embedding-table \
    --cross-entropy-loss \
    --pretrained-embedding-table  \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.01 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 6/8: CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=False
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=True::pretrained_embedding_table=False \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384 \
    --freeze-embedding-table \
    --cross-entropy-loss   \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.01 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 7/8: CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=True
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=True \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384  \
    --cross-entropy-loss \
    --pretrained-embedding-table  \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.01 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

# Experiment 8/8: CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=False
./sbatch.sh python train.py \
    --wandb-project-name respeller \
    --chkpt-save-dir /home/s1785140/respeller/exps/CrossEntropyTrainingV3::lrs=0.01::seeds=1337::freeze_embeddings=False::pretrained_embedding_table=False \
    --fastpitch-chkpt fastpitch/exps/halved_ljspeech_data_nospaces_noeos_pad_lowercase_nopunc/FastPitch_checkpoint_1000.pt \
    --input-type char \
    --symbol-set english_pad_lowercase_nopunc \
    --text-cleaners lowercase_no_punc \
    --use-mas \
    --cuda \
    --n-speakers 1 \
    --use-sepconv  \
    --respelling-len-modifier 0 \
    --nheads 2 \
    --num-layers 1 \
    --d-model 256 \
    --d-feedforward 512 \
    --dropout-inputs 0.0 \
    --dropout-layers 0.0 \
    --embedding-dim 384  \
    --cross-entropy-loss   \
    --gumbel-temp 2 2 1.0 \
    --batch-size 64 \
    --seed 1337 \
    --val-num-to-gen 32 \
    --softdtw-temp 1.0 \
    --dist-func l1 \
    --learning-rate 0.01 \
    --epochs 2000 \
    --val-log-interval 20 \
    --epochs-per-checkpoint 20

