# Assignment 1
## Ben Fuqua

### How to run code
First off, you need to know that when dataset 1 is reffered to in the code, it means the subscribed dataset and dataset 2 is the customer churn dataset. 

##### Step 1
Each dataset is ran seperatly in their respective files. "..._tune.py" is where I test all of the different hyperparameters and store those results in a csv called "..._results_d1.csv" or "..._results_d2.csv". 

##### Step 2
Then, I find which parameter combination is the best in the model_results.py file. Near the bottom of the file, you will see a cell that has the best parameters for that model for each of the models. 

##### Step 3
In the middle you will see an area where I loop through all of the results and pull out the best combination to get the best metric result. 

##### Step 4
Finally, in the "..._final.py" file is where I create the final version of my models and create the graphs that I will be using in my report. I am optimizing for precision so my graphs and hyperparameter choices will reflect this choice.

Here is the URL to get to my repo: https://github.com/cmbfuqua/GT_Machine_Learning/tree/main/Assignment-1




