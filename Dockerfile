FROM python:3.8

WORKDIR /src

COPY ./requirements.txt /src
COPY ./main.py /src

RUN pip install -r requirements.txt

EXPOSE 8000:8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
