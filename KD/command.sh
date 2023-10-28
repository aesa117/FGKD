# teacher
python3 -u teacher_train.py --dataset cora --teacher GCNII

# student baseline
python3 -u student_baseline.py --dataset cora --student GCNII

# KD student
python3 -u KD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01
python3 -u KD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01
python3 -u KD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10

# MustaD student
python3 -u MustaD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01
python3 -u MustaD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01
python3 -u MustaD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10

# CPF student
python3 -u CPF.py --dataset cora --teacher GCNII --student GCNII --lbd_embd 0.1
python3 -u CPF.py --dataset citeseer --teacher GCNII --student GCNII--lbd_embd 0.1
python3 -u CPF.py --dataset pubmed --teacher GCNII --student GCNII --lbd_embd 1

# Selector Model
python3 -u selector_pretrain.py --lr 0.01 --wd 0.001 --nlayer 5
<<<<<<< HEAD
python3 -u selector_pretrain.py --sage true --lr 0.01 --wd 0.001 --nlayer 5
=======
python3 -u selector_pretrain.py --sage True --lr 0.01 --wd 0.001 --nlayer 5
>>>>>>> 1388706359550c8c46e3f62952f85af9c96d18f0
