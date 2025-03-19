install:
	pip install \
	-r requirements/base.txt \
	-r requirements/dev.txt \
	-r requirements/local.txt

compile:
	rm -f requirements/*.txt
	pip-compile -o requirements/base.txt --strip-extras requirements/base.in
	pip-compile -o requirements/dev.txt --strip-extras requirements/dev.in
	pip-compile -o requirements/local.txt --strip-extras requirements/local.in

sync_peek:
	- pip-sync requirements/base.txt requirements/dev.txt requirements/local.txt -n

sync:
	pip-sync requirements/base.txt requirements/dev.txt requirements/local.txt

lint:
	python -m ruff check --fix

format:
	python -m ruff format

test:
	python -m pytest

typecheck:
	python -m mypy
