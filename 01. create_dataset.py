"""
코스피, 코스닥 수익율 데이터 가지고 time window로 짤라서 데이터 만들기
"""
import FinanceDataReader as fdr
from collections import deque
from datetime import date, time
import pandas as pd
import numpy as np
from argslist import *
import datetime
import re
import os

# create universe
if not os.path.exists('data/kospi_df.csv'):
    kospi_df = fdr.StockListing('KOSPI')  # 코스피  998
    kospi_df.Symbol = kospi_df.Symbol.astype('string').str.pad(width=6, side='left', fillchar='0')

    # screening
    kospi_df = kospi_df[kospi_df.ListingDate < START_DATE]  # 998  -> 480
    kospi_df.to_csv('kospi_df.csv')
else:
    kospi_df = pd.read_csv('data/kospi_df.csv')

if not os.path.exists('data/kosdaq_df.csv'):
    kosdaq_df = fdr.StockListing('KOSDAQ')  # 코스닥 1471
    kosdaq_df.Symbol = kosdaq_df.Symbol.astype('string').str.pad(width=6, side='left', fillchar='0')

    # screening
    kosdaq_df = kosdaq_df[kosdaq_df.ListingDate < START_DATE]  # 1470 -> 273
    kosdaq_df.to_csv('kosdaq_df.csv')
else:
    kosdaq_df = pd.read_csv('data/kosdaq_df.csv')
# nyse_df = fdr.StockListing('NYSE')   # 뉴욕거래소
# nasdaq_df = fdr.StockListing('NASDAQ') # 나스닥


if not os.path.exists('data/stocks.npy'):
    df = pd.concat([kospi_df, kosdaq_df])  # (733, 10)
    df.to_csv('stocks.npy')
else:
    df = pd.read_csv('data/stocks.npy')

"""
# 학습 데이터 초기 변수 선언 파트

START_DATE = '2001-01-01'
END_DATE = '2020-12-24'

# 샘플 수, 기간(휴일이 일정치 않으므로 보수적으로 245일), 데일리 수익율 <-- 어떤 종목인지 추가적으로 정보 기술 필요하므로
# 종가 데이터로만 판단하는 것으로 데이터 확정

샘플 수 : 20년 동안의 데이터에서 1년치 과거(타임 윈도우) 데이터를 제외한게 종목별로 년도별 학습데이터 생성 가능
753, 20*12, 245, 1 의 크기를 가지는 학습 데이터 생성 완료
"""

if not os.path.exists('data/x_dataset.npy'):
    x_dataset = np.zeros((753, 19 * TIME_WINDOW, DAYS_PERIOD, 1))
    df.reset_index(inplace=True)

    # saving csv of data frame for each stock
    for i, stock in df.iterrows():
        stock_df = fdr.DataReader(str(stock.Symbol).zfill(6), START_DATE, END_DATE)
        if len(stock_df) == LENGTH:
            months = pd.date_range(START_DATE, END_DATE, freq='BMS')
            for j, m in enumerate(months):
                if j - TIME_WINDOW >= 0:
                    current = datetime.datetime(*map(int, re.split('[^\d]', str(m))))
                    prev = current + datetime.timedelta(days=-365)
                    # 수익율의 상한과 하한을 +15,-15로 고정 (30, -22의 데이터 구간을 가지고 있었음)
                    profit = np.clip(stock_df.loc[prev:current]['Change'], a_max=0.15, a_min=-0.15)
                    profit = profit[:DAYS_PERIOD]
                    x_dataset[i, j - TIME_WINDOW, :, 0] = profit
    np.save('data/x_dataset.npy', x_dataset)
else:
    dataset = np.load('data/x_dataset.npy')


if not os.path.exists('data/y_dataset.npy'):
    y_dataset = np.zeros((753, 20*TIME_WINDOW, 2))
    months = pd.date_range(START_DATE, END_DATE, freq='BMS')

    for i, stock in df.iterrows():
        for j, m in enumerate(months):
            if j - TIME_WINDOW >= 0:
                y_dataset[i, j - TIME_WINDOW, 0] = stock.Symbol
                y_dataset[i, j - TIME_WINDOW, 1] = j
    np.save('data/y_dataset.npy', y_dataset)
else:
    y_dataset = np.load('data/y_dataset.npy')


if not os.path.exists('data/months.npy'):
    months = pd.date_range(START_DATE, END_DATE, freq='BMS')
    np.save('data/months.npy', months)





