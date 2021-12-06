
import uvicorn
from fastapi import FastAPI  
from ner_spacy import ner_spacy
# from word_embedding import embed_spacy
from transformer import summarize
import prometheus_client
from prometheus_client.core import CollectorRegistry
from prometheus_client import Summary, Counter, Histogram, Gauge
import time
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter
from starlette.responses import RedirectResponse
from pydantic import BaseModel


app = FastAPI(
    title = "NEW API",
    description = "NER extracts labels from the text"
    )

app.add_middleware(PrometheusMiddleware)
REDIRECT_COUNT = Counter("redirect_total", "Count of redirects", ["redirected_from"])
_INF = float("inf")

graphs = {}
graphs['c_ner_g'] = Counter('python_request_ner_total_g', 'The total number of (GET) ner processed requests')
graphs['h_ner_g'] = Histogram('python_request_ner_duration_seconds_g', 'Histogram for the (GET) ner duration in seconds.', buckets=(1, 2, 5, 6, 10, _INF))
graphs['c_sum_g'] = Counter('python_request_sum_total_g', 'The total number of (GET) summarization processed requests')
graphs['h_sum_g'] = Histogram('python_request_sum_duration_seconds_g', 'Histogram for the (GET) summarization duration in seconds.', buckets=(15, 30, 60, 120, 300, 360, 420, _INF))
graphs['c_ner_p'] = Counter('python_request_ner_total', 'The total number of ner processed requests')
graphs['h_ner_p'] = Histogram('python_request_ner_duration_seconds', 'Histogram for the ner duration in seconds.', buckets=(1, 2, 5, 6, 10, _INF))
graphs['c_sum_p'] = Counter('python_request_sum_total', 'The total number of summarization processed requests')
graphs['h_sum_p'] = Histogram('python_request_sum_duration_seconds', 'Histogram for the summarization duration in seconds.', buckets=(15, 30, 60, 120, 300, 360, 420, _INF))


class Text(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.get('/ner')
def ner_text(text):
    start_time = time.time()
    graphs['c_ner_g'].inc()
    output_spacy = list(ner_spacy(text))
    result = {"NER from Spacy": output_spacy}
    end_time = time.time()
    graphs['h_ner_g'].observe(end_time - start_time)
    return result

@app.get('/summary')
def summarize_text(text):
    start_time = time.time()
    graphs['c_sum_g'].inc()
    summary = summarize(text)
    result = {"Summary from transformers": summary}
    end_time = time.time()
    graphs['h_sum_g'].observe(end_time - start_time)
    return result

# @app.get("/test_metrics")
# def requests_count():
#     res = []
#     for k,v in graphs.items():
#         res.append(prometheus_client.generate_latest(v))
#     return res

@app.post("/summary")
def get_summary(text: Text):
    """Get summary from text"""
    # print(text.text)
    start_time = time.time()
    graphs['c_sum_p'].inc()
    summary = summarize(text.text)
    # summary = summarize(text)
    result = {"Summary from transformers": summary}
    end_time = time.time()
    graphs['h_sum_p'].observe(end_time - start_time)
    REDIRECT_COUNT.labels("sum_view").inc()
    return result

@app.post("/ner")
def get_ner(text: Text):
    """Get ner from text"""
    # print(text.text)
    start_time = time.time()
    graphs['c_ner_p'].inc()
    output_spacy = list(ner_spacy(text.text))
    # output_spacy = list(ner_spacy(text))
    result = {"NER from Spacy": output_spacy}
    end_time = time.time()
    graphs['h_ner_p'].observe(end_time - start_time)
    REDIRECT_COUNT.labels("ner_view").inc()
    return result


app.add_route("/metrics", handle_metrics)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
