default: all
.PHONY: all

all: minimal

.PHONY: minimal
minimal:
	python minimal.py data/test_pg6130.raw
