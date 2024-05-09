# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="zN774WuqAv4v"
# # Imports

# + id="RjqHoeVMS5Q8"
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
from pandas.plotting import autocorrelation_plot

# + [markdown] id="337bFOgLAjfs"
# # Variables

# + id="jWwOQXnXAaur"
data_folder_name = 'data'

# + [markdown] id="HodNv4I0AzKh"
# # Unpacking data

# + colab={"base_uri": "https://localhost:8080/"} id="Eer_4-ehRsDo" executionInfo={"status": "ok", "timestamp": 1711920019515, "user_tz": -120, "elapsed": 10, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="7a31187e-8f2e-4b29-cf61-4c3c8dac7491"
file_path = f'../{data_folder_name}/HICP-od-1997-co-miesiac.csv'

df_test = pd.read_csv(file_path)

print(df_test.head())
print(df_test.info())  # Not working - 1 column

# + colab={"base_uri": "https://localhost:8080/"} id="rDn5RdNDfEoQ" executionInfo={"status": "ok", "timestamp": 1711920019516, "user_tz": -120, "elapsed": 10, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="2bec87c5-52e4-43c2-c5fe-2b070321d8f1"
file_path = f'../{data_folder_name}/ECB Data Portal_20240312120629.csv'

df = pd.read_csv(file_path, parse_dates=['DATE'])

print(df.info())
print(df.head())

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="YpVLtBxU-_6V" executionInfo={"status": "ok", "timestamp": 1711919468656, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="e6adb438-aa92-4400-877a-66cc5f3406d7"
df = df.rename(columns={'HICP - Overall index (ICP.M.PL.N.000000.4.ANR)': 'HICP'})
df.head()

# + id="vlsg1FAnCoMx"
df.set_index('DATE', inplace=True)

# + [markdown] id="TwQFlSIIA3fg"
# # Exploring inflation value

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="Jhg1Hu3KBVFL" executionInfo={"status": "ok", "timestamp": 1711919472561, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="fe4d9f2c-385f-424e-b55b-f0d0581f036b"
df.describe()

# + [markdown] id="_nIOxSIwBbzk"
# ## Check for missing values

# + colab={"base_uri": "https://localhost:8080/"} id="rcZJcAl8BaZH" executionInfo={"status": "ok", "timestamp": 1711919473334, "user_tz": -120, "elapsed": 2, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="85134fcc-ac4c-479b-c463-d83fd728ad43"
df.isnull().sum()

# + [markdown] id="XYx_IVezCImv"
# ## Plot stuff

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="FI84kPfLCkms" executionInfo={"status": "ok", "timestamp": 1711919474917, "user_tz": -120, "elapsed": 521, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="14401ba8-60d7-4ca9-8d70-537039498beb"
df['HICP'].plot()

# + colab={"base_uri": "https://localhost:8080/", "height": 559} id="TA4iVRghBwh7" executionInfo={"status": "ok", "timestamp": 1711919476595, "user_tz": -120, "elapsed": 479, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="25a215db-3005-4fea-ee4f-6ac9b51b8ea1"
df['HICP'].rolling(window=12).mean().plot(figsize=(10, 6))  # 12-period moving average

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="gr8JsksaB9Lv" executionInfo={"status": "ok", "timestamp": 1711919477090, "user_tz": -120, "elapsed": 497, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="df8286a8-2d79-4340-9819-b5620867eace"
autocorrelation_plot(df['HICP'])

# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="G4OvzNbHqmM9" executionInfo={"status": "ok", "timestamp": 1711919477480, "user_tz": -120, "elapsed": 392, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="09b521fd-78b5-4271-a3b8-e5f728c5677b"
timespans = [
    {'start': '2007-08-01', 'end': '2009-06-30', 'caption': '2008 Crisis', 'color': 'green'},
    {'start': '2020-03-20', 'end': '2023-05-05', 'caption': 'Pandemic', 'color': 'blue'},
    {'start': '2022-02-24', 'end': '2024-03-12', 'caption': 'Russian Invasion', 'color': 'red'},
    {'start': '2009-10-01', 'end': '2011-01-01', 'caption': 'Eurozone Debt Crisis', 'color': 'pink'},
]

fig = px.line(df, y='HICP', title='HICP Overall Index Over Time with Highlighted Time Events')

for timespan in timespans:
    fig.add_vrect(x0=timespan['start'], x1=timespan['end'], annotation_text=timespan['caption'],
                  annotation_position="top left", fillcolor=timespan['color'], opacity=0.3, line_width=0)

events = [
    {'start': '2014-03-01', 'caption': 'Annexation of Crimea', 'color': 'red'},
    {'start': '2004-05-01', 'caption': 'EU Accession', 'color': 'blue'},
    {'start': '2019-07-01', 'caption': '500+', 'color': 'blue'},
]

for event in events:
    fig.add_vline(x=datetime.datetime.strptime(event['start'], "%Y-%m-%d").timestamp() * 1000,
                  annotation_text=event['caption'])

fig.show()

# + [markdown] id="f8UzbGTZA8IL"
# # Prediction
# -

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# + colab={"base_uri": "https://localhost:8080/", "height": 553} id="5QoNUSPKjTJw" executionInfo={"status": "ok", "timestamp": 1711919478950, "user_tz": -120, "elapsed": 438, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="af98324d-4607-4b69-c340-0e649e42bb93"
df = df.reset_index()
df_prophet = df.rename(columns={'DATE': 'ds', 'HICP': 'y'})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df_prophet)
months = 6
# Create a DataFrame for future dates. Let's say you want to predict the next 6 months
future_dates = model.make_future_dataframe(periods=months, freq='ME')

# Predict future inflation rates
forecast = model.predict(future_dates)

# Display the forecasted inflation rates along with confidence intervals
forecast.tail(months)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="yP2ncqilEvBO" executionInfo={"status": "ok", "timestamp": 1711919479332, "user_tz": -120, "elapsed": 386, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="9a52f3e8-7781-47a4-86b2-a3bd7f1ea2cf"
# Plot the forecast with uncertainty intervals
fig1 = plot_plotly(model, forecast)
fig1.show()

# Plot the forecast components
fig2 = plot_components_plotly(model, forecast)
fig2.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 564} id="hRifdSEWFubA" executionInfo={"status": "ok", "timestamp": 1711919481364, "user_tz": -120, "elapsed": 452, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="4d718ca5-cdc1-42cf-f535-e74262e9693a"
# Step 1: Identify the maximum date in the historical data
max_date = df_prophet['ds'].max()

# Step 2: Filter the forecast DataFrame to include only future dates
future_forecast = forecast[forecast['ds'] > max_date]

# Step 3: Plot only the future forecast
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(future_forecast['ds'], future_forecast['yhat'], label='Future Forecast')
ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='gray',
                alpha=0.2, label='Confidence Interval')

ax.set_title('Future Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Forecasted Value')
ax.legend()

plt.show()

# + [markdown] id="K-rfUG3fN-6n"
# # Merging datasets

# + colab={"base_uri": "https://localhost:8080/"} id="mN0xekOpN-Mn" executionInfo={"status": "ok", "timestamp": 1711920060579, "user_tz": -120, "elapsed": 266, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="094a72cf-f636-43b2-dff1-9c39879cf497"
file_path = f'../{data_folder_name}/ECB Data Portal_20240312120629.csv'

df = pd.read_csv(file_path, parse_dates=['DATE'])

print(df.info())
print(df.head())


# + id="txji5xbFb6_Q"
def excel_range_to_indices(excel_range):
    """
    Convert an Excel-style range into a list of (row_start, row_end, col_start, col_end) in zero-based indexing.

    Parameters:
    - excel_range: str, an Excel-style range, e.g., "A1:B2" or "D13:KF13".

    Returns:
    - A tuple of (row_start, row_end, col_start, col_end).
    """
    import re
    from openpyxl.utils.cell import column_index_from_string

    # Split the range into start and end parts
    start_cell, end_cell = excel_range.split(':')
    # Extract column letters and row numbers from both parts
    col_start, row_start = re.match(r"([A-Z]+)([0-9]+)", start_cell).groups()
    col_end, row_end = re.match(r"([A-Z]+)([0-9]+)", end_cell).groups()

    # Convert column letters to zero-based column indices
    col_start_idx = column_index_from_string(col_start) - 1
    col_end_idx = column_index_from_string(col_end) - 1
    # Convert row numbers to zero-based row indices
    row_start_idx = int(row_start) - 1
    row_end_idx = int(row_end) - 1

    return (row_start_idx, row_end_idx, col_start_idx, col_end_idx)


def add_excel_column_to_df(excel_file_path, sheet_name, excel_ranges, df, new_column_name, start_month):
    """
    Add data from specified ranges in an Excel file as a new column to a pandas DataFrame, aligning with a specified starting month.
    This version supports data that is split across multiple non-contiguous ranges and ensures the combined data does not exceed the DataFrame's length.

    Parameters:
    - excel_file_path: str, path to the Excel file.
    - sheet_name: int, sheet number (0-based indexing) or sheet name.
    - excel_ranges: list of str, list of Excel-style ranges, e.g., ["A1:D1", "E2:H2", "I3:Z3"].
    - df: pandas.DataFrame, DataFrame to add the new column to.
    - new_column_name: str, name of the new column.
    - start_month: str, the starting month for the new data in "YYYY-MM" format.

    Returns:
    - pandas.DataFrame, the DataFrame with the added column.
    """
    all_excel_data = []

    for excel_range in excel_ranges:
        # Convert Excel range to indices
        row_start, row_end, col_start, col_end = excel_range_to_indices(excel_range)
        #print(col_start)
        #print(col_end)

        # Read the specified range from the Excel sheet
        df_excel = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None,
                                 skiprows=row_start, nrows=row_end - row_start + 1,
                                 usecols=range(col_start, col_end + 1))

        # Flatten the data and add to the list
        excel_data = df_excel.values.flatten()
        all_excel_data.extend(excel_data)

    # Find the index of the starting month in the DataFrame
    start_index = df[df['month'] == start_month].index.min()
    # Calculate the maximum length of data that can be added
    max_length = len(df) - start_index
    # Truncate or pad the Excel data as necessary
    excel_data_aligned = [None] * start_index + all_excel_data[:max_length]
    if len(excel_data_aligned) < len(df):
        excel_data_aligned += [None] * (len(df) - len(excel_data_aligned))

    # Add the Excel data as a new column to the DataFrame
    df[new_column_name] = excel_data_aligned[:len(df)]

    return df


# + id="Y9FYVwZoW6W1"
csv_file_path = f'../{data_folder_name}/ECB Data Portal_20240312120629.csv'

df_csv = pd.read_csv(
    csv_file_path,
    usecols=[0, 2],  # Load only columns 0 and 2
    parse_dates=[0],  # Parse the first column as dates
    header=0  # Use the first row as header
)

# Convert the 'DATE' column to year and month format without the day
df_csv['DATE'] = df_csv['DATE'].dt.to_period('M')

# Rename columns to 'month' and 'HICP'
df_csv.columns = ['month', 'HICP']

# + id="blpM0Ct6Ze6j"
operations_map = {
    "BUDOWNICTWO": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D7:KF7'], "new_column_name": "Produkcja budowlano-montażowa (ceny stałe)"},
            {"excel_ranges": ['D12:KF12'], "new_column_name": "Mieszkania oddane do użytkowania"},
        ]
    },
    "BUDŻET PAŃSTWA": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D6:KE6'], "new_column_name": "Dochody budżetu państwa (w mln zł)"},
            {"excel_ranges": ['D7:KE7'], "new_column_name": "Wydatki budżetu państwa (w mln zł)"},
            {"excel_ranges": ['D7:KE7'], "new_column_name": "Wynik (saldo) budżetu państwa (w mln zł)"},
        ]
    },
    "HANDEL WEWN.": {
        "start_month": "2006-01",
        "operations": [
            {"excel_ranges": ['D7:HL7'], "new_column_name": "Sprzedaż detaliczna towarowa (ceny stałe)"},
            {"excel_ranges": ['D10:HL10'], "new_column_name": "Obroty w handlu detalicznym"},
        ]
    },
    "HANDEL ZAGR.": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D6:KE6'], "new_column_name": "Eksport towarów (ceny bieżące w mln zł)"},
            {"excel_ranges": ['D8:KE8'], "new_column_name": "Eksport towarów (ceny stałe B)"},
            {"excel_ranges": ['D9:KE9'], "new_column_name": "Import towarów (ceny bieżące w mln zł)"},
            {"excel_ranges": ['D11:KE11'], "new_column_name": "Import towarów (ceny stałe B)"},
            {"excel_ranges": ['D12:KE12'],
             "new_column_name": "Saldo obrotów towarowych handlu zagranicznego (w mln zł)"},
        ]
    },
    "KONIUNKTURA": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D6:KG6'], "new_column_name": "Bieżący wskaźnik ufności konsumenckiej (BWUK)"},
            {"excel_ranges": ['D7:KG7'], "new_column_name": "Wyprzedzający wskaźnik ufności konsumenckiej (WWUK)"},
            {"excel_ranges": ['D9:KG9'], "new_column_name": "Koniunktura - przetwórstwo przemysłowe"},
            {"excel_ranges": ['D10:KG10'], "new_column_name": "Koniunktura - budownictwo"},
            {"excel_ranges": ['D11:KG11'], "new_column_name": "Koniunktura - handel; naprawa pojazdów samochodowych"},
            {"excel_ranges": ['D12:KG12'], "new_column_name": "Koniunktura - transport i gospodarka magazynowa"},
            {"excel_ranges": ['D13:KG13'], "new_column_name": "Koniunktura - zakwaterowanie i gastronomia"},
            {"excel_ranges": ['D14:KG14'], "new_column_name": "Koniunktura - działalność finansowa i ubezpieczeniowa"},
            {"excel_ranges": ['D15:KG15'], "new_column_name": "Koniunktura - obsługa rynku nieruchomości"},
        ]
    },
    "PRZEMYSŁ": {
        "start_month": "2005-01",
        "operations": [
            {"excel_ranges": ['D7:HX7'], "new_column_name": "Produkcja sprzedana przemysłu ogółem (ceny stałe, B)"},
            {"excel_ranges": ['D13:HX13'], "new_column_name": "Produkcja - górnictwo i wydobywanie (B)"},
            {"excel_ranges": ['D19:HX19'], "new_column_name": "Produkcja - przetwórstwo przemysłowe (B)"},
            {"excel_ranges": ['D25:HX25'],
             "new_column_name": "Produkcja - wytwarzanie i zaopatrywanie w energię elektryczną, gaz, parę wodną i gorącą wodę (B)"},
            {"excel_ranges": ['D31:HX31'],
             "new_column_name": "Produkcja - dostawa wody; gospodarowanie ściekami i odpadami; rekultywacja (B)"},
            {"excel_ranges": ['D38:HX38'], "new_column_name": "Produkcja - dobra zaopatrzeniowe (B)"},
            {"excel_ranges": ['D40:HX40'], "new_column_name": "Produkcja - dobra inwestycyjne (B)"},
            {"excel_ranges": ['D42:HX42'], "new_column_name": "Produkcja - dobra konsumpcyjne trwałe (B)"},
            {"excel_ranges": ['D44:HX44'], "new_column_name": "dProdukcja - obra konsumpcyjne nietrwałe (B)"},
            {"excel_ranges": ['D46:HX46'], "new_column_name": "Produkcja - dobra związane z energią (B)"},
            {"excel_ranges": ['D48:HX48'], "new_column_name": "Nowe zamówienia w przemyślee (ceny bieżące, B)"},
        ]
    },
    "RYNEK PRACY": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D6:KF6'],
             "new_column_name": "Przeciętne zatrudnienie w sektorze przedsiębiorstwa (w w tys.)"},
            {"excel_ranges": ['D8:KF8'], "new_column_name": "Przeciętne zatrudnienie w sektorze przedsiębiorstwa (B)"},
            {"excel_ranges": ['D13:KF13'], "new_column_name": "Bezrobotni zarejestrowani (w tys.)"},
            {"excel_ranges": ['D15:KF15'], "new_column_name": "Bezrobotni zarejestrowani (B)"},
            {"excel_ranges": ['D18:AM18', 'AN17:JG17', 'JH16:KF16'],
             "new_column_name": "Stopa bezrobocia rejestrowanego (%)"},
        ]
    },
    "TRANSPORT": {
        "start_month": "2011-01",
        "operations": [
            {"excel_ranges": ['D6:FD6'],
             "new_column_name": "Przewozy ładunków w transporcie kolejowym (w mln tonokilometrów)"},
            {"excel_ranges": ['D8:FD8'], "new_column_name": "Przewozy ładunków w transporcie kolejowym (B)"},
            {"excel_ranges": ['D9:FD9'],
             "new_column_name": "Przewozy ładunków w transporcie kolejowym (od początku roku do końca okresu, w mln tonokilometrów)"},
            {"excel_ranges": ['D10:FD10'],
             "new_column_name": "Przewozy ładunków w transporcie kolejowym (od początku roku do końca okresu, A)"},
        ]
    },
    "WSKAŹNIKI CEN": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D7:KF7'], "new_column_name": "Wskaźniki cen skupu pszenicy (bez ziarna siewnego) (B)"},
            {"excel_ranges": ['D9:KF9'], "new_column_name": "Wskaźniki cen skupu żyta (bez ziarna siewnego) (B)"},
            {"excel_ranges": ['D11:KF11'], "new_column_name": "Wskaźniki cen skupu bydła (bez cieląt) (B)"},
            {"excel_ranges": ['D13:KF13'], "new_column_name": "Wskaźniki cen skupu trzody chlewnej (B)"},
            {"excel_ranges": ['D15:KF15'], "new_column_name": "Wskaźniki cen skupu mleka (B)"},
            {"excel_ranges": ['D16:KF16'],
             "new_column_name": "Relacje cen skupu żywca wieprzowego do cen żyta na targowiskach"},
            {"excel_ranges": ['D18:KF18'], "new_column_name": "Wskaźniki cen produkcji sprzedanej w przemyśle (B)"},
            {"excel_ranges": ['D21:KF21'], "new_column_name": "Wskaźniki cen w górnictwie i wydobywaniu (B)"},
            {"excel_ranges": ['D24:KF24'], "new_column_name": "Wskaźniki cen w przetwórstwie przemysłowym (B)"},
            {"excel_ranges": ['D27:KF27'],
             "new_column_name": "Wskaźniki cen w wytwarzaniu i zaopatrywaniu w energię elektryczną, gaz, parę wodną i gorącą wodę (B)"},
            {"excel_ranges": ['D30:KF30'],
             "new_column_name": "Wskaźniki cen w dostawie wody; gospodarowaniu ściekami i odpadami; rekultywacji (B)"},
            {"excel_ranges": ['D33:KF33'], "new_column_name": "Wskaźniki cen produkcji budowlano-montażowej (B)"},
            {"excel_ranges": ['D36:KF36'],
             "new_column_name": "Wskaźniki cen usług transportu i gospodarki magazynowej (B)"},
            {"excel_ranges": ['D39:KF39'], "new_column_name": "Wskaźniki cen usług telekomunikacji (B)"},
            {"excel_ranges": ['D42:KF42'], "new_column_name": "Wskaźniki cen towarów i usług konsumpcyjnych (B)"},
            {"excel_ranges": ['D45:KF45'], "new_column_name": "Wskaźniki cen - żywność i napoje bezalkoholowe (B)"},
            {"excel_ranges": ['D48:KF48'],
             "new_column_name": "Wskaźniki cen - napoje alkoholowe i wyroby tytoniowe (B)"},
            {"excel_ranges": ['D51:KF51'], "new_column_name": "Wskaźniki cen - odzież i obuwie (B)"},
            {"excel_ranges": ['D54:KF54'],
             "new_column_name": "Wskaźniki cen - użytkowanie mieszkania lub domu i nośniki energii (B)"},
            {"excel_ranges": ['D57:KF57'],
             "new_column_name": "Wskaźniki cen - wyposażenie mieszkania i prowadzenie gospodarstwa domowego (B)"},
            {"excel_ranges": ['D60:KF60'], "new_column_name": "Wskaźniki cen - zdrowie (B)"},
            {"excel_ranges": ['D63:KF63'], "new_column_name": "Wskaźniki cen - transport (B)"},
            {"excel_ranges": ['D66:KF66'], "new_column_name": "Wskaźniki cen - łączność (B)"},
            {"excel_ranges": ['D69:KF69'], "new_column_name": "Wskaźniki cen - rekreacja i kultura (B)"},
            {"excel_ranges": ['D72:KF72'], "new_column_name": "Wskaźniki cen - edukacja (B)"},
            {"excel_ranges": ['D74:KF74'], "new_column_name": "Wskaźniki cen transakcyjnych eksportu (A)"},
            {"excel_ranges": ['D75:KF75'], "new_column_name": "Wskaźniki cen transakcyjnych importu (A)"},
            {"excel_ranges": ['D76:KF76'], "new_column_name": "Terms of trade (A)"},
        ]
    },
    "WYNAGRODZ I ŚWIADCZ. SPOŁ": {
        "start_month": "2000-01",
        "operations": [
            {"excel_ranges": ['D6:KF6'],
             "new_column_name": "Przeciętne miesięczne nominalne wynagrodzenie brutto w sektorze przedsiębiorstwa (w zł)"},
            {"excel_ranges": ['D8:KF8'],
             "new_column_name": "Przeciętne miesięczne nominalne wynagrodzenie brutto w sektorze przedsiębiorstwa (B)"},
            {"excel_ranges": ['D10:KF10'],
             "new_column_name": "Przeciętne miesięczne realne wynagrodzenie brutto w sektorze przedsiębiorstwa (B)"},
            {"excel_ranges": ['D15:KF15'],
             "new_column_name": "Przeciętna miesięczna nominalna emerytura i renta brutto z pozarolniczego systemu ubezpieczeń społecznych (w zł)"},
            {"excel_ranges": ['D17:KF17'],
             "new_column_name": "Przeciętna miesięczna nominalna emerytura i renta brutto z pozarolniczego systemu ubezpieczeń społecznych (B)"},
            {"excel_ranges": ['D19:KF19'],
             "new_column_name": "Przeciętna miesięczna realna emerytura i renta brutto z pozarolniczego systemu ubezpieczeń społecznych (B)"},
            {"excel_ranges": ['D24:KF24'],
             "new_column_name": "Przeciętna miesięczna nominalna emerytura i renta rolników indywidualnych brutto (w zł)"},
            {"excel_ranges": ['D26:KF26'],
             "new_column_name": "Przeciętna miesięczna nominalna emerytura i renta rolników indywidualnych brutto (B)"},
            {"excel_ranges": ['D28:KF28'],
             "new_column_name": "Przeciętna miesięczna realna emerytura i renta rolników indywidualnych brutto (B)"},
        ]
    },
}

# + id="z1joBepUgvuu"
excel_file_path = f'../{data_folder_name}/wybrane_miesieczne_wskazniki_makroekonomiczne_cz_i.xlsx'
merged_df = df_csv.copy()

for sheet_name, info in operations_map.items():
    df = merged_df
    start_month = info["start_month"]
    for op in info["operations"]:
        add_excel_column_to_df(
            excel_file_path=excel_file_path,
            sheet_name=sheet_name,
            excel_ranges=op["excel_ranges"],
            df=df,
            new_column_name=op["new_column_name"],
            start_month=start_month
        )

# + colab={"base_uri": "https://localhost:8080/"} id="ZS43GcF9DoIj" executionInfo={"status": "ok", "timestamp": 1711920289780, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="45c27da8-9d9a-4bb9-9c4f-8fac27d2471f"
print(merged_df.info())
print(merged_df.head())

# + colab={"base_uri": "https://localhost:8080/"} id="3iphe9zwEYgD" executionInfo={"status": "ok", "timestamp": 1711920315481, "user_tz": -120, "elapsed": 267, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="e4a8312e-8212-44f3-f367-7b7718a7d3e7"
merged_df.replace(".", np.nan, inplace=True)
print(merged_df.info())
print(merged_df.head())

# + id="QxvD252mDY2L"
merged_df.to_csv('../data/merged.csv')

# + [markdown] id="t6RQyg9bbucy"
# # Data exploration

# + colab={"base_uri": "https://localhost:8080/", "height": 506} id="_fy3JLj79sMF" executionInfo={"status": "ok", "timestamp": 1711920322659, "user_tz": -120, "elapsed": 298, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="8a403477-5ee3-4a35-d78b-a247fe304816"
merged_df.drop(columns=['HICP']).describe()

# + id="zxM9Q_AhgGmP"
full_data = merged_df.copy()

# + id="zhHMke9WgEgb"
data_after_2018 = full_data[full_data['month'].dt.year > 2018]

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="HAAFt2zygIxn" executionInfo={"status": "ok", "timestamp": 1711920978034, "user_tz": -120, "elapsed": 405, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="771249ed-6562-41e0-bd40-96a16b9a6e4a"
data_after_2018.head()

# + colab={"base_uri": "https://localhost:8080/"} id="JmUbA9qiZ-8T" executionInfo={"status": "ok", "timestamp": 1711920988766, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="ff8ca196-a235-4ed1-8053-86b80ed9f543"
data_after_2018.info()


# + [markdown] id="Vmo-XynMbf79"
# ## Drop columns with no values

# + id="-qgyVwBFaLTZ"
def drop_cols_by_nan_fraction(df, max_nan_fraction=1.):
    columns_to_drop = []
    for col in df.columns:
        nan_val = df.loc[:, col].isna().any()
        if nan_val:
            nan_fraction = df.loc[:, col].isna().sum() / df.shape[0]
            if nan_fraction >= max_nan_fraction:
                columns_to_drop.append(col)

    result = df.drop(columns_to_drop, axis=1)
    return result


# + id="mKWTvJ0Vc5Vu"
data_after_2018_dropped = drop_cols_by_nan_fraction(data_after_2018)

# + colab={"base_uri": "https://localhost:8080/"} id="I64KHUPGbX1Y" executionInfo={"status": "ok", "timestamp": 1711921985650, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="c5fae794-8172-4316-8b3e-12a70eaff8c2"
data_after_2018_dropped.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="79uI4RQPd04-" executionInfo={"status": "ok", "timestamp": 1711921987356, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="97480e7f-654c-4582-8eae-327956a34134"
data_after_2018_dropped.tail()

# + colab={"base_uri": "https://localhost:8080/"} id="DwOv9GXNeX_c" executionInfo={"status": "ok", "timestamp": 1711921988255, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="eb31a799-0656-46fc-a991-6b761a2226c1"
data_after_2018_dropped = data_after_2018_dropped.iloc[:-2]
data_after_2018_dropped.info()

# + id="6zTM7TwXf1Po"
data_after_2018_dropped['Relacje cen skupu żywca wieprzowego do cen żyta na targowiskach'] = data_after_2018_dropped[
    'Relacje cen skupu żywca wieprzowego do cen żyta na targowiskach'].interpolate(method='linear')

# + colab={"base_uri": "https://localhost:8080/"} id="7CG1N15jf-rx" executionInfo={"status": "ok", "timestamp": 1711921994536, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="1dd2f6ed-b1e7-416e-a60b-8e72bce430b5"
data_after_2018_dropped.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 506} id="O1rVOFPngWSn" executionInfo={"status": "ok", "timestamp": 1711922053124, "user_tz": -120, "elapsed": 1103, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="01a68a3e-ca23-47e1-db88-b3c124575018"
data_after_2018_dropped.describe()
# -

data_after_2018_dropped['month'] = data_after_2018_dropped['month'].apply(lambda x: x.to_timestamp())
data_after_2018_dropped.to_csv('../data/filtered_data.csv')

# + [markdown] id="SWtl9XMkgtIF"
# ## Feature selection

# + id="fEOsayvgieu2"
y = data_after_2018_dropped["HICP"]
X = data_after_2018_dropped.drop(["HICP", "month"], axis=1)
X_HICP = data_after_2018_dropped["HICP"]
date_time = data_after_2018_dropped["month"]
# -

print(f'Shape of X: {X.shape}')
X.head()

print(f'Shape of y: {y.shape}')
y.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="0mv91-alioNa" executionInfo={"status": "ok", "timestamp": 1711922663058, "user_tz": -120, "elapsed": 661, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="6fb50351-562a-466e-c3ed-efd63a852afb"
print(f'Shape of date_time: {date_time.shape}')
date_time.head()
# -

print(f'Shape of X_HICP: {X_HICP.shape}')
X_HICP.head()

# ## Predictions using Darts library on HICP data

X_HICP_to_pred = pd.concat([X_HICP, date_time], axis=1)
X_HICP_to_pred.tail()

# +
from darts import TimeSeries

# Convert the DataFrame to a Darts TimeSeries object
X_HICP_to_pred = TimeSeries.from_dataframe(X_HICP_to_pred, 'month', 'HICP')

# split the series into training and validation sets

train, val = X_HICP_to_pred.split_after(pd.Timestamp('2023-05-01'))
# -

# ### NaiveSeasonal

val

# +
from darts.models import NaiveSeasonal

# Initialize the model
model = NaiveSeasonal(K=12)  # 12 months seasonality

# Fit the model
model.fit(X_HICP_to_pred)

# Make predictions
pred = model.predict(6)

print(f'Prediction values: {pred.values()}')

# Evaluate the model based on predictions and validation set
from darts.metrics import mape

print(f'Mean absolute percentage error: {mape(val, pred)}')
# -

# ## Feature selection using RFE and Linear Regression

# + colab={"base_uri": "https://localhost:8080/"} id="vDn5KGgEhmuP" executionInfo={"status": "ok", "timestamp": 1711922828794, "user_tz": -120, "elapsed": 1317, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="4ae4f10c-5301-4991-e371-a4d499e596a8"
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Example dataset for regression
df = pd.DataFrame(X)

# Estimator for feature selection in regression
model = LinearRegression()

# RFE model
selector = RFE(model, n_features_to_select=5)  # Select top 5 features
selector = selector.fit(df, y)

# Get the mask of selected features
selected_features = selector.support_

# Selected features DataFrame
selected_df = df.loc[:, selected_features]

# Print selected feature indices
print("Selected features indices:", selector.get_support(indices=True))
# -

# ## Prediction using Darts library on selected features using multiple models

df_to_pred = pd.concat([selected_df, date_time], axis=1)
df_to_pred.reset_index()
df_to_pred.head()

df_to_pred['month'] = df_to_pred['month'].apply(lambda x: x.to_timestamp())

# +
from darts import TimeSeries

# Convert the DataFrame to a Darts TimeSeries object
series = TimeSeries.from_dataframe(df_to_pred, 'month', df_to_pred.columns.tolist()[:-1])

# Plot the TimeSeries
series.plot()

# Train-test split
train, val = series.split_after(pd.Timestamp('2023-11-01'))
# -

# ### NaiveSeasonal

# +
from darts.models import NaiveSeasonal

# Initialize the model
model = NaiveSeasonal(K=12)  # 12 months seasonality

# Fit the model
model.fit(train)

# Make predictions
pred = model.predict(4)

print(f'Prediction values: {pred.values()}')
# -

# ### ARIMA

# +
from darts.models import ARIMA

# Initialize the model
model = ARIMA(p=1, d=1, q=1)

# Fit the model
model.fit(train)

# Make predictions
pred = model.predict(4)

print(f'Prediction values: {pred.values()}')

# + colab={"base_uri": "https://localhost:8080/", "height": 553} id="-W4R0I-woo-p" executionInfo={"status": "ok", "timestamp": 1711924438086, "user_tz": -120, "elapsed": 897, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="954de6db-4f77-478f-a2b4-64deb7642715"
df_prophet = data_after_2018_dropped.reset_index()
df_prophet = data_after_2018_dropped.rename(columns={'month': 'ds', 'HICP': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.to_timestamp())

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df_prophet)
months = 6
# Create a DataFrame for future dates. Let's say you want to predict the next 6 months
future_dates = model.make_future_dataframe(periods=months, freq='M')

# Predict future inflation rates
forecast = model.predict(future_dates)

# Display the forecasted inflation rates along with confidence intervals
forecast.tail(months)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="fVxbu1IfpkEn" executionInfo={"status": "ok", "timestamp": 1711924466702, "user_tz": -120, "elapsed": 983, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="a02f2735-fd69-4704-ec2b-a8c21d5f0976"
# Plot the forecast with uncertainty intervals
fig1 = plot_plotly(model, forecast)
fig1.show()

# Plot the forecast components
fig2 = plot_components_plotly(model, forecast)
fig2.show()

# + id="yZit4c-mnk3-"
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

# Assuming df is your time series dataframe with a datetime index and 'y' is the target column
y = data_after_2018_dropped['HICP']

# Analyze ACF and PACF, possibly to determine p and q
# acf_values = acf(y)
# pacf_values = pacf(y)

# Fit ARIMA model
# Model parameters (p, d, q) should be chosen based on your time series analysis
model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()

# Forecast
forecasts = model_fit.forecast(steps=5)  # for example, forecasting 5 steps ahead

# Display the forecasted values
print(forecasts)


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AleXWcWAjWZM" executionInfo={"status": "ok", "timestamp": 1711922864994, "user_tz": -120, "elapsed": 278, "user": {"displayName": "Dominik Cio\u0142czyk", "userId": "00022494602056967109"}} outputId="9ab11fef-a9f2-4aa1-b9c8-7ee4574758bd"
df.iloc[:, selector.get_support(indices=True)]

# + colab={"base_uri": "https://localhost:8080/"} id="4vnShuVSgx8g" executionInfo={"status": "ok", "timestamp": 1711910901833, "user_tz": -120, "elapsed": 427, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="0aa8b4f2-1e76-4f34-d17c-4a5e6db86f2a"

statistical_summary = data.describe()

# Correlation Matrix for selected variables
corr_matrix = data.corr()

statistical_summary, corr_matrix

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="yjHHvg8y_e1S" executionInfo={"status": "ok", "timestamp": 1711910901833, "user_tz": -120, "elapsed": 6, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="fb2856c1-370d-4592-f0df-426e20746137"
# Correlation Matrix for selected indicators
# Zrobiłem, żeby było więcej do docxa jak coś xdd
corr_matrix = data.drop(columns=['HICP']).corr()

corr_matrix

# + colab={"base_uri": "https://localhost:8080/"} id="Y-mchC9iA_Y1" executionInfo={"status": "ok", "timestamp": 1711910901833, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="218381f5-b33b-4f28-ee72-0a8d1201ae36"
# To też dla Zygmunt xdd

# Przekształcenie macierzy korelacji w postać szeregów danych
corr_series = corr_matrix.unstack()

# Filtrowanie korelacji równych 1, wyświetlanie tylko jednej strony korelacji, unikalne wartości, unikalne korelacje
filtered_corr_series = corr_series[
    (corr_series != 1) & (corr_series.index.get_level_values(0) < corr_series.index.get_level_values(1))]
filtered_corr_series = filtered_corr_series[~filtered_corr_series.duplicated()]

# Znalezienie n największych wartości korelacji
n = 10  # liczba największych wartości korelacji do wyświetlenia
top_corr = filtered_corr_series.nlargest(n)

# Wyświetlenie wyników
print(f"\n{n} największych unikalnych korelacji (pomijając korelacje równą 1 oraz drugą stronę korelacji):")
for idx, val in top_corr.items():
    print(f"Korelacja między {idx[0]} i {idx[1]} wynosi {val}")

# + colab={"base_uri": "https://localhost:8080/", "height": 490} id="IKaxhPg0hnt7" executionInfo={"status": "error", "timestamp": 1711910902690, "user_tz": -120, "elapsed": 860, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="03007559-536d-4fce-f4f7-7894ecee7886"
hicp_correlations = corr_matrix['HICP'].drop('HICP')

# Sorting the correlations to get the most and least correlated variables with HICP
hicp_correlations_sorted_absolute = hicp_correlations.abs().sort_values(ascending=False)

# Displaying the sorted correlations by absolute value
hicp_correlations_sorted_absolute

# + colab={"base_uri": "https://localhost:8080/", "height": 506} id="Odo8HHOWdtl7" executionInfo={"status": "ok", "timestamp": 1711910985570, "user_tz": -120, "elapsed": 1660, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="7b5870c0-883e-419b-acca-fee01e603c78"
import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'month' to datetime format for plotting
data['month'] = pd.to_datetime(data['month'], errors='coerce')

# Filter out columns with more than 20% missing values for a cleaner correlation analysis
filtered_columns = data.columns[data.isnull().mean() < 0.05]
filtered_data = data[filtered_columns]

# Statistical Summary of key variables
statistical_summary = filtered_data.describe()

# Correlation Matrix for selected variables
corr_matrix = filtered_data.corr()

# Plotting correlation heatmap
#plt.figure(figsize=(12, 10))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Correlation Heatmap of Selected Variables')
#plt.tight_layout()

#plt.show(), statistical_summary
statistical_summary

# + colab={"base_uri": "https://localhost:8080/"} id="XEEdu9eX2dDC" executionInfo={"status": "ok", "timestamp": 1711910989428, "user_tz": -120, "elapsed": 2, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="5283a3ea-2322-46b9-9915-3cd01fc45c70"
hicp_correlations_filtered = corr_matrix['HICP'].drop('HICP')

# Sorting the correlations to get the most and least correlated variables with HICP
hicp_correlations_sorted_absolute_filtered = hicp_correlations_filtered.abs().sort_values(ascending=False)

# Displaying the sorted correlations by absolute value
hicp_correlations_sorted_absolute_filtered

# + colab={"base_uri": "https://localhost:8080/"} id="yJw4kmkuHDFz" executionInfo={"status": "ok", "timestamp": 1711910998934, "user_tz": -120, "elapsed": 2828, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="7bcd00dd-1a2c-495b-a2e5-025f9c643486"
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'df' is your DataFrame and it's already sorted chronologically
# 'date_column' is the column with the dates or months
# 'target' is the column you're trying to predict

# Splitting the data - let's say the last 'n' rows are your test set
n = 12  # For example, last 12 months as test set
train_df = merged_df[:-n]
test_df = merged_df[-n:]

# Separating features and target
X_train = train_df.drop(['HICP', 'month'], axis=1)
y_train = train_df['HICP']
X_test = test_df.drop(['HICP', 'month'], axis=1)
y_test = test_df['HICP']

# Initialize and train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', missing=np.nan)  # XGBoost can handle NaN values
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse}")

# + colab={"base_uri": "https://localhost:8080/"} id="jZXEMBihHqaM" executionInfo={"status": "ok", "timestamp": 1711910998934, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="5f7caa90-14e7-404f-cd4b-3210cf76eaab"
predictions

# + colab={"base_uri": "https://localhost:8080/", "height": 607} id="aaTXYn4HHn_M" executionInfo={"status": "ok", "timestamp": 1711910999943, "user_tz": -120, "elapsed": 607, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="7e1cf0ef-8950-4177-95e4-ecf5f09558c2"
test_dates = pd.date_range(start="2023-03-02", periods=n, freq='MS')

plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test, label='Actual', marker='o')
plt.plot(test_dates, predictions, label='Predicted', linestyle='--', marker='x')
plt.title('Actual vs Predicted Values (Test Set)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 376} id="WsxrXAkQJ5W3" executionInfo={"status": "error", "timestamp": 1711910999943, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="33d89ccd-ce44-4992-a394-066d2a03566c"
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(merged_df['HICP'], model='additive')
fig = decomposition.plot()
plt.show()

# + id="dZhy2uusKZU4"
# Splitting the data - let's say the last 'n' rows are your test set
n = 12  # For example, last 12 months as test set
train_df = merged_df[:-n]
test_df = merged_df[-n:]

# Separating features and target
X_train = train_df.drop(['HICP', 'month'], axis=1)
y_train = train_df['HICP']
X_test = test_df.drop(['HICP', 'month'], axis=1)
y_test = test_df['HICP']

# + colab={"base_uri": "https://localhost:8080/"} id="VcOsnQNxKGdL" executionInfo={"status": "ok", "timestamp": 1711911450335, "user_tz": -120, "elapsed": 481, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="6e632c52-06ca-411d-caac-6af7b9983549"
from statsmodels.tsa.arima.model import ARIMA

# Example: ARIMA(1,1,1)
model = ARIMA(merged_df['HICP'], order=(1, 1, 1))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="wGHh0AncKKzZ" executionInfo={"status": "ok", "timestamp": 1711911452220, "user_tz": -120, "elapsed": 608, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="a570c363-cfc0-4bd7-b741-74d4cb124105"
# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot the forecast alongside the original series
pd.concat([merged_df['HICP'], forecast]).plot(figsize=(10, 5))

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="x7YQ0ub439F4" executionInfo={"status": "ok", "timestamp": 1711911466023, "user_tz": -120, "elapsed": 516, "user": {"displayName": "Szymon Budziak", "userId": "04173189432736408336"}} outputId="8a31a297-b707-4439-eb68-966a1c9a7c9a"
merged_df.head()

# + [markdown] id="tJn5mk602teH"
# 2 major components
