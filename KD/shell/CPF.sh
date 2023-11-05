for data in cora citeseer pubmed
do
    for teacher in GCNII GAT GraphSAGE
    do
        for student in GCNII GAT GraphSAGE PLP
            python3 -u CPF.py --dataset ${data} --teacher ${teacher} --student ${student} --lbd_embd 0.1 --mask 20
    done
done