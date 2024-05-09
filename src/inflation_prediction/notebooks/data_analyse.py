#%% md
# # Imports
#%%
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
from pandas.plotting import autocorrelation_plot
#%% md
# # Variables
#%%
data_folder_name = 'data'
#%% md
# # Unpacking data
#%%
file_path = f'../{data_folder_name}/HICP-od-1997-co-miesiac.csv'

df_test = pd.read_csv(file_path)

print(df_test.head())
print(df_test.info())  # Not working - 1 column
#%%
file_path = f'../{data_folder_name}/ECB Data Portal_20240312120629.csv'

df = pd.read_csv(file_path, parse_dates=['DATE'])

print(df.info())
print(df.head())
#%%
df = df.rename(columns={'HICP - Overall index (ICP.M.PL.N.000000.4.ANR)': 'HICP'})
df.head()
#%%
df.set_index('DATE', inplace=True)
#%% md
# # Exploring inflation value
#%%
df.describe()
#%% md
# ## Check for missing values
#%%
df.isnull().sum()
#%% md
# ## Plot stuff
#%%
df['HICP'].plot()
#%%
df['HICP'].rolling(window=12).mean().plot(figsize=(10, 6))  # 12-period moving average
#%%
autocorrelation_plot(df['HICP'])
#%%
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
#%% md
# # Merging datasets
#%%
file_path = f'../{data_folder_name}/ECB Data Portal_20240312120629.csv'

df = pd.read_csv(file_path, parse_dates=['DATE'])

print(df.info())
print(df.head())
#%%
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
#%%
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
#%%
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
            {"excel_ranges": ['D44:HX44'], "new_column_name": "Produkcja - dobra konsumpcyjne nietrwałe (B)"},
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
#%%
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
#%%
print(merged_df.info())
print(merged_df.head())
#%%
merged_df.replace(".", np.nan, inplace=True)
print(merged_df.info())
print(merged_df.head())
#%% md
# # Data exploration
#%%
full_data = merged_df.copy()
#%%
data_after_2018 = full_data[full_data['month'].dt.year > 2018]
#%%
data_after_2018.head()
#%%
data_after_2018.info()
#%% md
# ## Drop columns with no values
#%%
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
#%%
data_after_2018_dropped = drop_cols_by_nan_fraction(data_after_2018)
#%%
data_after_2018_dropped.info()
#%%
data_after_2018_dropped.tail()
#%%
data_after_2018_dropped = data_after_2018_dropped.iloc[:-2]
data_after_2018_dropped.info()
#%%
data_after_2018_dropped['Relacje cen skupu żywca wieprzowego do cen żyta na targowiskach'] = data_after_2018_dropped[
    'Relacje cen skupu żywca wieprzowego do cen żyta na targowiskach'].interpolate(method='linear')
#%%
data_after_2018_dropped.info()
#%%
data_after_2018_dropped.describe()
#%%
data_after_2018_dropped['month'] = data_after_2018_dropped['month'].apply(lambda x: x.to_timestamp())
data_after_2018_dropped.to_csv('../data/filtered_data.csv')