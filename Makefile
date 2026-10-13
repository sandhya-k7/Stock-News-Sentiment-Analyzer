install:
	pip install -r requirements.txt
test:
	pytest -q
finetune:
	python -m src.sentiment_finetune
app:
	streamlit run app.py
