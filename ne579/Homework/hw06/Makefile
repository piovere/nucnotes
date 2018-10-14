report:
	pdflatex report.tex
	biber report
	pdflatex report.tex

clean:
	rm *.run.xml
	rm *.aux
	rm *.bbl
	rm *.blg
	rm *.log
	rm -rf __pycache__

graphics:
	. activate 579 && python hw06.py

images: hw06.py
	. activate 579 && python hw06.py
