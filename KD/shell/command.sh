# teacher
python3 -u teacher_train.py --dataset cora --teacher GCNII

# student baseline
python3 -u student_baseline.py --dataset cora --student GCNII
python3 -u student_baseline.py --dataset cora --student GCNII --mustad true

# KD student
python3 -u KD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01 --mask 20
python3 -u KD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01 --mask 20
python3 -u KD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10 --mask 20

# MustaD student
python3 -u MustaD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01 --mask 20
python3 -u MustaD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01 --mask 20
python3 -u MustaD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10 --mask 20

# CPF student
python3 -u CPF.py --dataset cora --teacher GCNII --student GCNII --lbd_embd 0.1 --mask 20
python3 -u CPF.py --dataset citeseer --teacher GCNII --student GCNII--lbd_embd 0.1 --mask 20
python3 -u CPF.py --dataset pubmed --teacher GCNII --student GCNII --lbd_embd 1 --mask 20

# Selector Model pretrain
python3 -u selector_pretrain.py --lr 0.01 --wd 0.001 --margin 0.3 --ms1 500 --ms2 750 --gm 0.1 --nlayer 5