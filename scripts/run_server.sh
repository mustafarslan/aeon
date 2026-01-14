#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/shell
uvicorn aeon_py.server:app --host 0.0.0.0 --port 8000 --reload
