# MLOps-LLMOps

This repo contains the results of several practical exercises on Machine Learning Operations and Large Language Models Ops:

* `Ariadna_HV_modelDeployment_Practical.ipynb`: Notebook containing all the development of an NLP model to predict the sentiment of the text of reviews. The model is trained with the dataset `Sports_and_Outdoors_5.json`, which can be downloaded from https://drive.google.com/file/d/17AD9Hhw4kEoe3ORcJKGf9MfJIGHleBav/view (entire dataset.tar). 

    The project includes a full pipeline for the training and testing phase, allowing for easier experimentation with different parameters and versions of the model. 

    The different experiments were **tracked and logged using MLflow**, and the best model was registered. The code interactions with MLflow are also included in the notebook, and **screenshots of the MLflow User Interface** are included in the document `Screenshots_AriadnaHV.pdf`. 
    
    Finally, the notebook also contains the scripts corresponding to some **experiments done with LLMs**, in particular via the library `LangChain`. The scripts were also logged in **MLflow** to keep track of them and for comparison of different traces. Again, **screenshots** of these interactions are provided in the document `Screenshots_AriadnaHV.pdf`. For obvious reasons, the `.env` file used for these calls is not shared.
    
* `utils.py`: To allow for the notebook to remain efficiently organized, the code for several functions (in partiular those used for the preprocessing of the data) is kept in a separate  file the content of which is imported into the notebook. 

* `hello_fastapi.py` is the file containing the Python scripts used for building APIs, and integrating them within FastAPI, for higher performance and ease of use. Again, numerous screenshots of the interaction with FastAPI are included in `Screenshots_AriadnaHV.pdf`.

* `Screenshots_AriadnaHV.pdf`: document containing images (screenshots) of interactions with the various MLOps and LLMOps apps used in this module, mostly MLflow and FastAPI. 

* Finally, some of the "light" files created while running the sentiment prediction model have been included (other files, such as the dictionary of preprocessed, have not due to their very large size):
    * `metrics.json` contains the metrics of the last run.
    * `model.pkl` is a copy of the trained model.
