PROJ=vardl
TESTS=tests


static:
	flake8 $(PROJ)

test:
	pytest -v $(TESTS)

coverage:
	pytest -v --cov $(PROJ) $(TESTS)

