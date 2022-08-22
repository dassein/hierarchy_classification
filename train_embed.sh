rm -rf embed
rm -rf cluster
rm -rf pretrains
rm -rf models
rm -rf csv_logs
rm -rf tb_logs
rm -rf visualize
python train_embed.py --stage 0
python train_embed.py --stage 1
python train_embed.py --stage 2
python train_embed.py --stage 3
python train_embed.py --stage 4
python train_embed.py --stage 5
python visualize.py