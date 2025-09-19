# Makefile for LaTeX project

# --- Variables ---
MAIN_TEX = main.tex
PDF = $(MAIN_TEX:.tex=.pdf)
LATEX = pdflatex -interaction=nonstopmode -file-line-error
BIBTEX = bibtex

# --- Targets ---

all: $(PDF)

# Rule to compile the LaTeX file to PDF
# Runs latex, bibtex, and then latex twice to resolve references
$(PDF): $(wildcard *.tex) $(wildcard */*.tex) references.bib
	@echo "--- Compiling LaTeX (Pass 1) ---"
	@$(LATEX) $(MAIN_TEX)
	@echo "--- Compiling Bibliography ---"
	@$(BIBTEX) $(MAIN_TEX:.tex=)
	@echo "--- Compiling LaTeX (Pass 2) ---"
	@$(LATEX) $(MAIN_TEX)
	@echo "--- Compiling LaTeX (Pass 3) ---"
	@$(LATEX) $(MAIN_TEX)
	@echo "PDF generated: $(PDF)"

view: $(PDF)
	xdg-open $(PDF) &

clean:
	@rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	@rm -f $(PDF)
	@echo "Cleaned up generated files."

.PHONY: all view clean