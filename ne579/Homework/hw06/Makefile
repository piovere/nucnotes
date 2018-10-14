report: images/first_four_latent_variables.png	images/pls_weights_and_correlation.png	images/rmse_vs_pca_loadings.png images/pls_vs_pcr_loadings.png images/rmse_vs_number_components.png bibliography.bib
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
	. activate 579 && python hw05.py

images/first_four_latent_variables.png	images/pls_weights_and_correlation.png	images/rmse_vs_pca_loadings.png images/pls_vs_pcr_loadings.png images/rmse_vs_number_components.png: hw05.py
	. activate 579 && python hw05.py
