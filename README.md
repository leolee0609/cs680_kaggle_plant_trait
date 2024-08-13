### Instruction
Please follow these simple steps to run the model:
0. Use the submission on Learn rather than github if you can. The github version doesn't include precomputed image embedding, which can be time consuming if you obtain it yourself.
1. Upload and unpack the source_code compression file in your working directory
2. Open catboost.ipynb
3. Run and import the dependent packages in the first cell
4. Add the training set train.csv and test set test.csv to the ./data folder in the working directory
* If you want to change the dataset, please run get_embedding.ipynb and add the image folders to the ./data folder
5. Run everything else to train CatBoost models and predict plant traits based on the image embeddings and transformed tabular data