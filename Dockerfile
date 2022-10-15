FROM python:3.8

WORKDIR /src

COPY ./requirements.txt /src
COPY ./Run_Model_API.py /src
COPY ./Linear_Regression_Model.pkl /src

RUN pip install -r requirements.txt

EXPOSE 8000:8000

CMD ["uvicorn", "Run_Model_API:app", "--reload", "--host", "0.0.0.0"]
