# for data in cora citeseer pubmed
# do
#     for model in GCNII GAT GraphSAGE PLP
#     do
#         python3 -u student_baseline.py --dataset ${data} --student ${model}
#     done
# done

# # mustaD setting
# for data in cora citeseer pubmed
# do
#     for model in GCNII GAT GraphSAGE
#     do
#         python3 -u student_baseline.py --dataset ${data} --student ${model} --mustad true
#     done
# done

for data in cora citeseer pubmed
do
    python3 -u student_baseline.py --dataset ${data} --student GAT --mustad true
done