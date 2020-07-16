import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import requests
import pandas as pd
from pylab import plt, mpl
import matplotlib.pyplot as plt
import lxml
def main():

    st.title('Curt App')
    st.subheader('Making Investing less intimidating')

#####get ticker data
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = requests.get(url).text
    df = pd.read_html(data)
    df = df[0]


    web_data = requests.get('https://en.wikipedia.org/wiki/NASDAQ-100#Changes_in_2020').text
    nasdaq_df = pd.read_html(web_data)
    nasdaq_df = nasdaq_df[2]


    def load_data_history(tickers, start_date):
        ticker_data = yf.Tickers(tickers)
        data = ticker_data.history(period=start_date)
        for ticker in tickers:   
            fig = go.Figure()
            fig = px.line(data, x=data.index, y=data['Close'][ticker])
            fig.update_xaxes(title='Date')
            fig.update_yaxes(title='Price')
            st.subheader(ticker + "'s Price History")
            st.plotly_chart(fig)
    

    def daily_returns(tickers, start_date):
        ticker_data = yf.Tickers(tickers)
        data = ticker_data.history(period=start_date)
        for ticker in tickers:
            daily_returns_data = data['Close'][ticker].pct_change()
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            daily_returns_data.plot.hist(bins=60)
            ax.set_xlabel('Daily Returns')
            ax.set_ylabel('Percent')
            ax.set_title( ticker + ' Daily Returns Data')
    #plt.show()
            st.write(fig)

    def moving_averages(tickers, start_date):
        ticker_data = yf.Tickers(tickers)
        data = ticker_data.history(period=start_date)
        for ticker in tickers:
            data['SMA1'] = data['Close'][ticker].rolling(window=50).mean()
            data['SMA2'] = data['Close'][ticker].rolling(window=200).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'][ticker], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA1'], mode='lines', name='50 day avg'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA2'], mode='lines', name='200 day avg'))
            st.subheader(ticker + ' 50-Day Moving Average Vs. 200-Day Moving Average')
            st.write(fig)

    def plot_returns(tickers):
        data = yf.Tickers(tickers)
        df = data.history(period=start_date)
        returns = df['Close'].resample('M').ffill().pct_change()
        returns1 = (returns + 1).cumprod()
        fig = go.Figure()
        for column in returns1:
            fig.add_trace(go.Scatter(x=returns1.index, y=returns1[column], mode='lines', name=column))
            fig.update_layout(
                title='Cumulative Returns on $1 Investment'
        )
        st.write(fig)

    index = st.sidebar.selectbox('Choose an Index: ', ['S&P 500', 'NASDAQ', 'Search for Ticker'])
    dropdown_option = st.sidebar.selectbox('Pick a Chart', ['Price History', 'Short-Vs-Long-Term Moving Avg', 'Daily Returns', 'Cumulative Returns'])
    start_date = st.sidebar.selectbox('Interval', ['1y', '3y', '5y', '10y', 'max'])
    if index == 'S&P 500':
        dropdown = st.sidebar.multiselect('S&P Tickers', df['Symbol'])
        results = st.sidebar.checkbox('Get Results')
        if results:
            if dropdown_option == 'Price History':
                data = load_data_history(dropdown, start_date)
            elif dropdown_option == 'Daily Returns':

        #st.subheader(dropdown + "'s daily returns over " + start_date)
                data = daily_returns(dropdown, start_date)
            elif dropdown_option == 'Short-Vs-Long-Term Moving Avg':
        #st.subheader(dropdown + "'s 50-day Moving Average Vs. 200-Day Moving Average")
                data = moving_averages(dropdown, start_date)
            elif dropdown_option == 'Cumulative Returns':
                stocks = dropdown
                ticker_list = []
        #number_tickers = st.sidebar.number_input('Specify the number of tickers')
                for stock in stocks:
                    ticker_list.append(stock)
                plot_returns(ticker_list)


    elif index == 'NASDAQ':
        dropdown = st.sidebar.multiselect('Nasdaq Tickers', nasdaq_df['Ticker'])
        results = st.sidebar.checkbox('Get Results')
        if results:
            if dropdown_option == 'Price History':
                data = load_data_history(dropdown, start_date)
            elif dropdown_option == 'Daily Returns':
                data = daily_returns(dropdown, start_date)
            elif dropdown_option == 'Short-Vs-Long-Term Moving Avg':
                data = moving_averages(dropdown, start_date)
            elif dropdown_option == 'Cumulative Returns':
                stocks = dropdown
                ticker_list = []
        #number_tickers = st.sidebar.number_input('Specify the number of tickers')
                for stock in stocks:
                    ticker_list.append(stock)
                plot_returns(ticker_list)
if __name__ == '__main__':
    main()