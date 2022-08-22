rm -rf cluster
mkdir cluster
rm -rf pretrains
rm -rf models
rm -rf csv_logs
rm -rf tb_logs
python train.py --stage 1
python train.py --stage 2
python train.py --stage 3
python train.py --stage 4
python train.py --stage 5