from fastapi import FastAPI, Query
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
import warnings
import re
import calendar
from datetime import datetime

warnings.filterwarnings("ignore")

templates = Jinja2Templates(directory="./templates")
app = FastAPI(title="ArXiv Paper Search Engine")

es = Elasticsearch({
    "scheme": "http",
    "host": "host.docker.internal",
    "port": 9200
}, max_retries=30, retry_on_timeout=True, request_timeout=30)

if not es.ping():
    raise ValueError("Elasticsearch connection failed")
else:
    print("Connected to Elasticsearch")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def parse_date_term(term: str):
    date_formats = [
        "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
        "%m/%Y", "%m-%Y", "%m.%Y",
        "%Y/%m", "%Y-%m", "%Y.%m",
        "%Y"
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(term, fmt)
            if fmt == "%Y":
                start = datetime(dt.year, 1, 1)
                end = datetime(dt.year, 12, 31)
            elif fmt in ["%Y/%m", "%Y-%m", "%Y.%m"]:
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                start = datetime(dt.year, dt.month, 1)
                end = datetime(dt.year, dt.month, last_day)
            elif fmt in ["%m/%Y", "%m-%Y", "%m.%Y"]:
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                start = datetime(dt.year, dt.month, 1)
                end = datetime(dt.year, dt.month, last_day)
            else:
                start = end = dt
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None, None


@app.get("/health", summary="Health Check")
def health():
    return {"message": "OK"}


@app.get("/", summary="Home Page")
def home_page(request):
    result = "Type a keyword or year"
    return templates.TemplateResponse('home.html', context={'request': request, 'result': result})


@app.get("/search", summary="Search in arxiv_index")
def search_arxiv(term: str = Query(..., min_length=2), size: int = 20, from_: int = 0):
    start_date, end_date = parse_date_term(term)

    if start_date and end_date:
        # Tìm theo khoảng ngày nếu parse được ngày tháng
        query = {
            "query": {
                "range": {
                    "published_date": {
                        "gte": start_date,
                        "lte": end_date
                    }
                }
            },
            "size": size,
            "from": from_
        }
    else:
        # Tìm kết hợp term chính xác và full-text fuzzy
        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "term": {
                                "arxiv_id.keyword": {
                                    "value": term,
                                    "boost": 10
                                }
                            }
                        },
                        {
                            "term": {
                                "journal_ref.keyword": {
                                    "value": term,
                                    "boost": 6
                                }
                            }
                        },
                        {
                            "term": {
                                "categories.keyword": {
                                    "value": term,
                                    "boost": 6
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "authors": {
                                    "query": term,
                                    "boost": 5
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": term,
                                "fields": [
                                    "title^4",
                                    "abstract^2",
                                    "comment"
                                ],
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "size": size,
            "from": from_
        }

    res = es.search(index="arxiv_index", body=query)
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    total = res["hits"]["total"]["value"]

    return {"results": hits, "total": total, "size": size, "from": from_}
