PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

graphics:
	. activate powers-luhn_cosc528_fall-2018_project01 && python 01-jrpl-Dataset_examination.py

report: graphics
	pdflatex project1.tex
	find . -type f -name "*.aux" -delete
	find . -type f -name "*.log" -delete

install:
	conda create -y -n powers-luhn_cosc528_fall-2018_project01
	conda install -y --name powers-luhn_cosc528_fall-2018_project01 -c conda-forge --file requirements.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

uninstall:
	conda env remove -y -n powers-luhn_cosc528_fall-2018_project01
