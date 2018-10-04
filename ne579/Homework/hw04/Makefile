pdf: paper.tex paper.pdf utilities.py Homework\ 04\ Clean.ipynb
	jupyter nbconvert --to script Homework\ 04\ Clean.ipynb
	mv Homework\ 04\ Clean.py appendix/appendix.py
	cp utilities.py appendix/
	sed -E "s|get_ipython\(\).run_line_magic\('[a-z]*', '([a-z]+).py'\)|from \1 import *|" < appendix/appendix.py > appendix/appendix2.py
	sed -E "s|(get_ipython\(\).run_line_magic\('matplotlib'[.]*)|# \1|" < appendix/appendix2.py > appendix/appendix3.py
	mv appendix/appendix3.py appendix/appendix.py
	rm appendix/appendix2.py
	pdflatex paper.tex
	biber paper
	pdflatex paper.tex
	pdflatex paper.tex
	rm *.aux
	rm *.log
	rm paper.bcf
	rm paper.out
	rm paper.run.xml
	rm paper.blg
	rm paper.bbl
