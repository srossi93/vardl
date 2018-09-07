PYTHON := python3
MAIN := ./class-exp.py

DATASET := spam
FOLD := 0

experiment: blm uninformative heuristic xavier-normal orthogonal lsuv



blm:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2

uninformative:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2

heuristic:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2

xavier-normal:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2

orthogonal:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2

lsuv:
	@$(PYTHON) $(MAIN) with fold=$(FOLD) dataset=$(DATASET) init_strategy=$@
	@sleep 2
