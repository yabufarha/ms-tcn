OS := $(shell uname -s)


.PHONY:	setup
setup:
	ln -sf poetry.$(OS).lock poetry.lock


.PHONY:	install
install:
	poetry lock --no-update
	poetry install


.PHONY:	fmt
fmt:
	poetry run black .
	poetry run isort .


.PHONY:	lint
lint:
	poetry run ruff check .


.PHONY: typecheck
typecheck:
	poetry run pyright .
