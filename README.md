# Election-Analyser
    A Election analyzer is ML project Made for Hacksagon 2026
    Our machine learning models are trained on extensive historical and survey data to map the complex relationships between voter identity and political preference.
    **Election Analyser** is a machine learning-powered predictive engine and API designed to decode this complexity. Built with FastAPI and scikit-learn, this application takes granular demographic data and translates it into actionable political intelligence for state elections in India (currently supporting Bihar and Maharashtra).
## Features
    * **FastAPI Backend:** A fast, modern Python API using Pydantic for data validation.
    * **State-Specific Endpoints:** Dedicated prediction routes and UI templates for Bihar and Maharashtra.
    * **Web Interface:** Serves static frontend files and HTML templates for a seamless user experience.
## Tech Stack
    * **Backend:** FastAPI, Uvicorn, Python
    * **Data & ML:** Pandas, NumPy, Scikit-learn
    * **Frontend:** HTML, CSS, JavaScript, Jinja2
## Installation
    ```Bash
    git clone https://github.com/Existencee3/Election-Analyser.git
    cd Election-Analyser
    python3 -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
    uvicorn app:app --reload --port 5000
    ```
