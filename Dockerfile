# 
FROM python:3.9

#
EXPOSE 8000

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
#RUN cp -r /root/nltk_data /usr/local/share/nltk_data 

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]