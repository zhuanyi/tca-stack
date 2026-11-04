UV ?= uv
RUN = $(UV) run

.PHONY: setup data simulate tca figures notes test lint reproduce

setup:
	$(UV) sync

data:
	$(RUN) tca ust data
	$(RUN) tca fx data

simulate:
	$(RUN) tca ust simulate
	$(RUN) tca fx simulate

tca:
	$(RUN) tca ust tca
	$(RUN) tca fx tca
	$(RUN) tca db build

figures:
	$(RUN) tca ust figures
	$(RUN) tca fx figures

notes:
	$(RUN) python notes/04_algo_wheel/analysis.py
	$(RUN) python notes/05_fx_fixing/analysis.py
	$(RUN) python notes/06_rfq_dealer_count/analysis.py

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

reproduce: data simulate tca figures notes
