for data in cora citeseer pubmed
do
    for teacher in GCNII GAT GraphSAGE
    do
        python3 -u CPF.py --dataset ${data} --teacher ${teacher} --student PLP --lbd_embd 0.1 --mask 20
    done
done