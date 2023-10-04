# FKD
Feature selection based Knowledge Distillation implementation

- KD
  - dir
    - KD_student dir : trained student checkpoint files using Knowledge Distillation Methods
    - data dir : npz(dataset file) & files related to dataset loading
    - models dir : configuration yaml files & GNN, PLP, MLP models for training
    - teachers dir : trained student checkpoint files
    - utils dir : will delete
  - files
    - KD.py : Common Knowledge Distillation using feature selection
    - MustaD.py : MustaD KD using feature selection
    - CPF.py : CPF KD using feature selection
    - mask.py : feature mask generate & update & selector network training
    - teacher_train.py : teacher models training
    - student_baseline.py : student baseline models training
    - command.sh : command examples
