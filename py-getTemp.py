
url = "https://tianqi.2345.com/Pc/GetHistory"

headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}

import requests
import pandas as pd

def craw_table(year, month):
    #enter year and month, will return temperature data
    params={
            "areaInfo[areaId]": 58362,
            "areaInfo[areaType]": 2,
            "date[year]": year,
            "date[month]": month}

    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()["data"]
    df = pd.read_html(data)[0]
    return df

df_list=[]
    
for year in range(2013,2023):
    for month in range(1,13):
        print("getting",year,month)
        df = craw_table(year,month)
        df_list.append(df)
        
pd.concat(df_list).to_excel("SH 10y temp.xlsx", index=False)