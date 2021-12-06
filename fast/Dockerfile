FROM tiangolo/uvicorn-gunicorn:python3.8

RUN mkdir /fastapi_folder

COPY requirements.txt /fastapi_folder

WORKDIR /fastapi_folder

RUN pip install -r requirements.txt 
RUN python -m spacy download en
COPY . /fastapi_folder

EXPOSE 80

CMD ["python", "main.py"]