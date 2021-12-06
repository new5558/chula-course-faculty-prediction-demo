FROM python:3.7
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install sentencepiece
RUN pip install thai2transformers==0.1.2 --no-dependencies
RUN pip install matplotlib==3.5.0
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]