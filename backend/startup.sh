#!/bin/bash
# Azure App Service startup command for FastAPI
# This file is referenced in Azure portal > Configuration > Startup Command

cd /home/site/wwwroot
pip install -r requirements.txt --quiet
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
