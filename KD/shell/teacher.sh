# for data in cora citeseer pubmed
# do
#     for model in GCNII GAT GraphSAGE
#     do
#         python3 -u teacher_train.py --dataset ${data} --teacher ${model}
#     done
# done

for data in cora citeseer pubmed
do
    for model in GCNII GAT GraphSAGE
    do
        python3 -u teacher_train.py --dataset ${data} --teacher ${model}
    done
done