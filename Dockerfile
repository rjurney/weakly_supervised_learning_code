FROM  continuumio/anaconda3:latest

# Copy this directory over, only to be over-ridden by the docker-compose volume of same
COPY . /weakly_supervised_learning_code
WORKDIR /weakly_supervised_learning_code

# Install Python dependencies and setup Jupyter without authentication
RUN pip install -r requirements.in && \
    jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Run Jupyter
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
