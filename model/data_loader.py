import pandas as pd
import requests
import zipfile
from io import BytesIO

def load_drybean_from_uci():
    url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"

    r = requests.get(url)
    r.raise_for_status()

    z = zipfile.ZipFile(BytesIO(r.content))
    xlsx_name = [f for f in z.namelist() if f.endswith(".xlsx")][0]

    with z.open(xlsx_name) as f:
        df = pd.read_excel(f)

    return df
