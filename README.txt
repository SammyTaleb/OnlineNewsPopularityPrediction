###### JUPYTER NOTEBOOK ######
To view the jupyter notebook, you can open the .ipynb file or run the API and go to the url:
http:/localhost:5000/notebook

###### TO RUN THE API #######
Start by cloning the github into your own directory. As the model files .pkl are very heavy, I was not able to push them on github.
Please run the file 'models.py' to get the .pkl files containing each model.
You can then open a IPython Console, move to the 'API Flask' Directory and run the command:
python app.py go to the localhost URL to see the API.

###### POWERPOINT ####### 
The Powerpoint presentation, gathers all important information and process details of the dataset analysis 
and the models building.

###### CONCLUSION OF THE PREDICTION TASKS ######
The regression did not perform well on this dataset. Classification tasks performed better. 
Best models were Random Forest and Gradient Boosting classifiers with tuned parameters. These models have been deployed in the API.
Best accuracy obtained : 67.1%, Best AUC : 73%
