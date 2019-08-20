PROJ=vardl
TESTS=tests


static:
	flake8 $(PROJ)

test:
	pytest -v $(TESTS)

coverage:
	pytest -v --cov $(PROJ) $(TESTS)

git-clean: git-clean-check
	@git clean -dfX
	@echo Done

git-clean-check:
	@git clean -dfXn
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
