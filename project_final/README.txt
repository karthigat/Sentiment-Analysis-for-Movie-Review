----README-----

Files attached:

1. multinominal_nb.py - Multinominal Naive Baysian model
2. svm_classification.py - SVM model
3. streamlit_app_mnb.py - Streamlit app for multinominal model
4. streamlit_app_svm.py - Streamlit app for SVM model
5. IMDB Dataset - Dataset for training and testing models
6. requirements.txt - libraries to be installed

Folder attached:

Movie_images - folder contains
	1. 110 movie files
	2. Streamlit_movie Dataset.csv - dataset for streamlit app

Instruction to run Multinomial Navie Baysian:

Step 1: Run multinominal_nb.py(it will generate movie dataset.csv)
Step 2: Run streamlit_app_mnb.py(Dashboard will be open in the browser)
Step 3: Check the browser for output

Instruction to run SVM:

Step 1: Run svm_classification.py(it will generate movie dataset.csv)
Step 2: Run streamlit_app_svm.py(Dashboard will be open in the browser)
Step 3: Check the browser for output

File generated:

1. From multinominal_nb.py following files are generated:
	a. heatmap_mnb.png
	b. wordcloud_positive_mnb.png
	c. wordcloud_negative_mnb.png
	d. Movie image folder -> movie_dataset_nb.csv

2. From svm_classification.py following files are generated:
	a. heatmap_svm.png
	a. wordcloud_positive_svm.png
	c. wordcloud_negative_svm.png
	d. Movie image folder -> movie_dataset_svm.csv
