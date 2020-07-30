import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import scipy.stats as scs
import scipy.optimize as sco
import requests
import pandas as pd
from pylab import plt, mpl
import matplotlib.pyplot as plt
import lxml
import ffn 
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import discrete_allocation, DiscreteAllocation
import numpy as np
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
#%matplotlib inline
def main():

    #st.title('Curt App')
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
        if len(tickers) > 1 and type(tickers) != str:
            ticker_data = yf.Tickers(tickers)
            data = ticker_data.history(period=start_date)
            for ticker in tickers:   
                fig = go.Figure()
                fig = px.line(data, x=data.index, y=data['Close'][ticker])
                fig.update_xaxes(title='Date')
                fig.update_yaxes(title='Price')
                st.subheader(ticker + "'s Price History")
                st.plotly_chart(fig)
        elif len(tickers) == 1 and type(tickers) != str:
            ticker = tickers[0]
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)   
            fig = go.Figure()
            fig = px.line(data, x=data.index, y=data['Close'])
            fig.update_xaxes(title='Date')
            fig.update_yaxes(title='Price')
            st.subheader(ticker + "'s Price History")
            st.plotly_chart(fig)
        elif type(tickers) == str:
            ticker = tickers
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)
            fig = go.Figure()
            fig = px.line(data, x=data.index, y=data['Close'])
            fig.update_xaxes(title='Date')
            fig.update_yaxes(title='Price')
            st.subheader(ticker + "'s Price History")
            st.plotly_chart(fig)

    

    def daily_returns(tickers, start_date):
        if len(tickers) > 1 and type(tickers) != str:
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
                st.write(fig)

        elif len(tickers) == 1 and type(tickers) != str:
            ticker = tickers[0]
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)
            daily_returns_data = data['Close'].pct_change()
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            daily_returns_data.plot.hist(bins=60)
            ax.set_xlabel('Daily Returns')
            ax.set_ylabel('Percent')
            ax.set_title( ticker + ' Daily Returns Data')
            st.write(fig)

        elif type(tickers) == str:
            ticker = tickers
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)
            daily_returns_data = data['Close'].pct_change()
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            daily_returns_data.plot.hist(bins=60)
            ax.set_xlabel('Daily Returns')
            ax.set_ylabel('Percent')
            ax.set_title( ticker + ' Daily Returns Data')
            st.write(fig)

    def moving_averages(tickers, start_date):
        if len(tickers) > 1 and type(tickers) != str:
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
        elif len(tickers) == 1 and type(tickers) != str:
            ticker = tickers[0]
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)
            data['SMA1'] = data['Close'].rolling(window=50).mean()
            data['SMA2'] = data['Close'].rolling(window=200).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA1'], mode='lines', name='50 day avg'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA2'], mode='lines', name='200 day avg'))
            st.subheader(ticker + ' 50-Day Moving Average Vs. 200-Day Moving Average')
            st.write(fig)

        elif type(tickers) == str:
            ticker = tickers
            ticker_data = yf.Ticker(ticker)
            data = ticker_data.history(period=start_date)
            data['SMA1'] = data['Close'].rolling(window=50).mean()
            data['SMA2'] = data['Close'].rolling(window=200).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA1'], mode='lines', name='50 day avg'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA2'], mode='lines', name='200 day avg'))
            st.subheader(ticker + ' 50-Day Moving Average Vs. 200-Day Moving Average')
            st.write(fig)

            

    def plot_returns(tickers):
        if len(tickers) > 1 and type(tickers) != str:
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
        elif len(tickers) == 1 and type(tickers) != str:
            ticker = tickers[0]
            data = yf.Ticker(ticker)
            df = data.history(period=start_date)
            returns = df['Close'].resample('M').ffill().pct_change()
            returns1 = (returns + 1).cumprod()
            returns1 = returns1.to_frame('Close')
            fig = px.line(returns1, x=returns1.index, y=returns1['Close'])
            st.write(fig)

        elif type(tickers) == str:
            ticker = tickers
            data = yf.Ticker(ticker)
            df = data.history(period=start_date)
            returns = df['Close'].resample('M').ffill().pct_change()
            returns1 = (returns + 1).cumprod()
            returns1 = returns1.to_frame('Close')
            fig = px.line(returns1, x=returns1.index, y=returns1['Close'], title='Cumulative Returns on $1 Investment')
            st.write(fig)

    def expected_r(tickers, start_date):
        today = pd.datetime.today()
        if start_date == '1y':
            delta = today - pd.DateOffset(years=1)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '3y':
            delta = today - pd.DateOffset(years=3)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '5y':
            delta = today - pd.DateOffset(years=5)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '10y':
            delta = today - pd.DateOffset(years=10)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == 'max':
            delta = today - pd.DateOffset(years=30)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')

        prices = ffn.get(tickers, start=delta)
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        st.write(cleaned_weights)
        metrics = ef.portfolio_performance(verbose=True)
        st.write('Expected Return: {:.2f}'.format(metrics[0]))
        st.write('Annual Volatility: {:.2f}'.format(metrics[1]))
        st.write('Sharpe Ratio {:.2f}'.format(metrics[2]))

    def port_ret(weights):
        return np.sum(rets.mean() * weights) * 252

    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

    def min_func_sharpe(weights):
        return -port_ret(weights)/port_vol(weights)
        
        
    def expected_r2(tickers, start_date):
        today = pd.datetime.today()
        if start_date == '1y':
            delta = today - pd.DateOffset(years=1)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '3y':
            delta = today - pd.DateOffset(years=3)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '5y':
            delta = today - pd.DateOffset(years=5)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '10y':
            delta = today - pd.DateOffset(years=10)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == 'max':
            delta = today - pd.DateOffset(years=30)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')

        prices = ffn.get(tickers, start=delta)
        noa = len(tickers)
        global rets
        rets = np.log(prices / prices.shift(1))
        #rets.hist(bins=40, figsize=(10,8))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        bnds = tuple((0,1) for x in range(noa))
        eweights = np.array(noa * [1. / noa,])
        opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
        st.write("The expected return is: {:.2f}".format(port_ret(opts['x'].round(3))))
        st.write("The expected volatility is: {:.2f}".format(port_vol(opts['x'].round(3))))
        st.write("The Shapre Ratio is: {:.2f}".format(port_ret(opts['x']/port_vol(opts['x']))))
        st.subheader("How to best allocate the portfolio to maximize the return:")
        i = 0
        for x in opts['x']:
            st.write(tickers[i] + ": " + str(x.round(2)))
            i = i + 1

        prets = []
        pvols = []
        for p in range(2500):
            weights = np.random.random(noa)
            weights /= np.sum(weights)
            prets.append(port_ret(weights))
            pvols.append(port_vol(weights))
        prets = np.array(prets)
        pvols = np.array(pvols)
        #optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
        #cons = ({'type': 'eq', 'fun': lambda x: port_ret(x)- tret}, {'type': 'eq', 'fun': lambda x: np.sum(x)-1})

        #bnds = tuple((0,1) for x in weights)

        #trets = np.linspace(0.05, 0.3, 50)
        #tvols = []
        #for tret in trets:
        #    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
        #    tvols.append(res['fun'])
        #tvols = np.array(tvols)

        #fig, ax = plt.subplots()
        #im = plt.scatter(pvols, prets, c=prets/pvols, marker='.', alpha=0.8, cmap='coolwarm')
        #plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)
        #plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
        #plt.xlabel('Expected Volatility')
        #plt.ylabel('Expected Return')
        #fig.colorbar(im, label='Sharpe ratio')
        #st.write(fig)

        fig, ax = plt.subplots()
        im= plt.scatter(pvols, prets, c=prets/pvols, marker='o', cmap='coolwarm')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        fig.colorbar(im, label='Sharpe Ratio')
        st.write(fig)
    def asset_allocation(tickers, start_date):
        today = pd.datetime.today()
        if start_date == '1y':
            delta = today - pd.DateOffset(years=1)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '3y':
            delta = today - pd.DateOffset(years=3)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '5y':
            delta = today - pd.DateOffset(years=5)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == '10y':
            delta = today - pd.DateOffset(years=10)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        elif start_date == 'max':
            delta = today - pd.DateOffset(years=30)
            delta = delta.date()
            delta = delta.strftime('%Y-%m-%d')
        prices = ffn.get(tickers, start=delta)
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        latest_prices = discrete_allocation.get_latest_prices(prices)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount)
        allocation, leftover = da.lp_portfolio()
        st.subheader('Asset Allocation breakdown: ')
        st.write(allocation)
        st.write("Funds remaining: ${:.2f}".format(leftover))


    index = st.sidebar.selectbox('Choose an Index: ', ['S&P 500', 'NASDAQ', 'Search for Ticker'])
    dropdown_option = st.sidebar.selectbox('Pick a Chart', ['Price History', 'Short-Vs-Long-Term Moving Avg', 'Daily Returns', 'Cumulative Returns', 'Expected Returns', 'Asset Allocation'])
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
                st.markdown("The 50-day moving average exceeding the 200-day moving average is a buy signal.")
        #st.subheader(dropdown + "'s 50-day Moving Average Vs. 200-Day Moving Average")
                data = moving_averages(dropdown, start_date)
            elif dropdown_option == 'Cumulative Returns':
                stocks = dropdown
                ticker_list = []
        #number_tickers = st.sidebar.number_input('Specify the number of tickers')
                for stock in stocks:
                    ticker_list.append(stock)
                plot_returns(ticker_list)
            elif dropdown_option == 'Expected Returns':
                data = expected_r2(dropdown, start_date)
            elif dropdown_option == 'Asset Allocation':
                amount = st.sidebar.number_input('Enter the amount you wish to invest')
                button = st.sidebar.button('Click for Results')
                if button:
                    data = asset_allocation(dropdown, start_date)


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
            elif dropdown_option == 'Expected Returns':
                data = expected_r2(dropdown, start_date)
            elif dropdown_option == 'Asset Allocation':
                amount = st.sidebar.number_input('Enter the amount you wish to invest')
                button = st.sidebar.button('Click for Results')
                if button:
                    data = asset_allocation(dropdown, start_date)
    
    elif index == 'Search for Ticker':
        dropdown = st.sidebar.text_input('enter a series of tickers e.g. DOCU MSFT AMZN')
        dropdown = dropdown.split()
        if dropdown_option == 'Price History':
            data = load_data_history(dropdown, start_date)
        elif dropdown_option == 'Daily Returns':
            data = daily_returns(dropdown, start_date)
        elif dropdown_option == 'Short-Vs-Long-Term Moving Avg':
            data = moving_averages(dropdown, start_date)
        elif dropdown_option == 'Cumulative Returns':
        #number_tickers = st.sidebar.number_input('Specify the number of tickers')  
            plot_returns(dropdown)
        elif dropdown_option == 'Expected Returns':
            st.markdown("""This page illustrates how to best allocate your portfolio
            by maximizing the expected return and minimzing the amount volatility associated with that return. 
            Note, Sharpe Ratio of 1 or higher is considered very good.""")
            data = expected_r2(dropdown, start_date)
        elif dropdown_option == 'Asset Allocation':
            amount = st.sidebar.number_input('Enter the amount you wish to invest')
            button = st.sidebar.button('Click for Results')
            if button:
                st.markdown("recommends the number of shares to buy from each stcok chosen, based on the investment amount.")
                data = asset_allocation(dropdown, start_date)

    
if __name__ == "__main__":
    main()