FROM jupyter/datascience-notebook 

# Update to locked version of python pacakges
COPY requirements.txt .
RUN conda install -y --file requirements.txt
