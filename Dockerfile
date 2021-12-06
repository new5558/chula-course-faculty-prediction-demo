FROM python:3.7
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install sentencepiece
RUN pip install thai2transformers==0.1.2 --no-dependencies
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]