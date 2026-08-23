SHELL := /bin/zsh
TEXBIN ?= $(HOME)/Library/TinyTeX/bin/universal-darwin
TEXS := $(shell find cs1001/lectures cs4880/lectures -name '*.tex' | sort)

.PHONY: all clean list

all:
	@set -e; for tex in $(TEXS); do \
	  course=$${tex%%/*}; \
	  base=$${tex##*/}; \
	  name=$${base%.tex}; \
	  mkdir -p $$course/pdfs build/$$course/$$name; \
	  echo "Building $$course/pdfs/$$name.pdf"; \
	  (cd $$(dirname $$tex) && LANG=C LC_ALL=C PATH="$(TEXBIN):$$PATH" latexmk -quiet -pdf -interaction=nonstopmode -halt-on-error -outdir=../../build/$$course/$$name $$base); \
	  cp build/$$course/$$name/$$name.pdf $$course/pdfs/$$name.pdf; \
	done

list:
	@printf '%s\n' $(TEXS)

clean:
	rm -rf build
	find cs1001/pdfs cs4880/pdfs -name '*.pdf' -delete 2>/dev/null || true
