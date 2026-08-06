FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY theme.py .
COPY streamlit_app.py .

CMD [ "streamlit", "run", "streamlit_app.py" ]
