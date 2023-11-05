for data in cora citeseer pubmed
do
    for teacher in GCNII GAT GraphSAGE
    do
        for student in GCNII GAT GraphSAGE PLP
        do
            python3 -u KD.py --dataset ${data} --teacher ${teacher} --student ${student} --lbd_pred 1 --lbd_embd 0.01 --mask 20
        done
    done
done