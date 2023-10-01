# KD student
python -u KD/KD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01
python -u KD/KD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01
python -u KD/KD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10

# MustaD student
python -u KD/MustaD.py --dataset cora --teacher GCNII --student GCNII --lbd_pred 1 --lbd_embd 0.01
python -u KD/MustaD.py --dataset citeseer --teacher GCNII --student GCNII --lbd_pred 0.1 --lbd_embd 0.01
python -u KD/MustaD.py --dataset pubmed --teacher GCNII --student GCNII --lbd_pred 100 --lbd_embd 10

# CPF student
python -u KD/CPF.py --dataset cora --teacher GCNII --student GCNII --lbd_embd 0.1
python -u KD/CPF.py --dataset citeseer --teacher GCNII --student GCNII--lbd_embd 0.1
python -u KD/CPF.py --dataset pubmed --teacher GCNII --student GCNII --lbd_embd 1