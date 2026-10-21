PYTHON ?= python
PIP ?= pip
SEED ?= 42
PORT ?= 8501
DB := data/funnel.duckdb

.PHONY: setup data demo test eval app screenshot clean

setup:  ## install dependencies
	$(PIP) install -r requirements.txt

data:  ## generate ~1.5M synthetic sessions + planted anomalies, load into DuckDB
	$(PYTHON) scripts/generate_events.py --seed $(SEED)

$(DB):
	$(PYTHON) scripts/generate_events.py --seed $(SEED)

demo: $(DB)  ## top insights + executive summary in the terminal, HTML report in reports/
	$(PYTHON) -m funnel_insights.demo

test:  ## offline test suite (no key, no network)
	$(PYTHON) -m pytest

eval:  ## precision / recall / detection delay vs planted anomalies (6 seeds + sensitivity sweep)
	$(PYTHON) -m funnel_insights.evaluate

app: $(DB)  ## Streamlit dashboard
	streamlit run app.py --server.port $(PORT)

screenshot:  ## capture docs/screenshot.png from a running app (make app in another terminal)
	$(PYTHON) scripts/screenshot.py --url http://localhost:$(PORT) --height 2150

clean:
	rm -rf data reports .pytest_cache
