PYTHON := python3
MAIN := ./class-depth-exp.py

DATASET := spam
FOLD := 0
GPU := 2
experiment: blm lsuv uninformative heuristic xavier-normal orthogonal



blm:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)

uninformative:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)

heuristic:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)

xavier-normal:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)

orthogonal:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)

lsuv:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@ device=cuda:$(GPU)
