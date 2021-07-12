# Person attribute Classification and re-identification - Unitn's course project

In this project, we used deep learning method for attributes classification and re-identification tasks. For each
task, we tried to combine several methods to improve performance and also experiment un different network architectures. 

code:
* re_id.ipynb : using to train network with triplet loss
* test_re_id_final.ipynb : using to generate final result with reranking task
* test_re_id_map.ipynb: using map metric to evaluate different methods
* training.ipynb: training file
* prepare_data.ipynb: 
	- Data augmentation
	- split training and validation set
* classification_evaluate.ipynb: evaluate classification task
	- evaluate different models on validation set
	- generate classification result
* identification_evaluate.ipynb: build embedding vector and evaluate re-identification task
* model.py: models for classification task

result:
* reid test.txt : result for re-identification task
* classification_test.csv : result for classification task
	
