import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to read and process the CSV file
def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = df["Date"].str.replace(".000", "")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    df = df[df.High != df.Low]
    df.set_index("Date", inplace=True)
    return df

# Function to calculate TotalSignal
def total_signal(df, current_candle):
    current_pos = df.index.get_loc(current_candle)

    # Long entry conditions
    c1 = df['High'].iloc[current_pos] > df['Close'].iloc[current_pos]
    c2 = df['Close'].iloc[current_pos] > df['High'].iloc[current_pos - 2]
    c3 = df['High'].iloc[current_pos - 2] > df['High'].iloc[current_pos - 1]
    c4 = df['High'].iloc[current_pos - 1] > df['Low'].iloc[current_pos]
    c5 = df['Low'].iloc[current_pos] > df['Low'].iloc[current_pos - 2]
    c6 = df['Low'].iloc[current_pos - 2] > df['Low'].iloc[current_pos - 1]

    if c1 and c2 and c3 and c4 and c5 and c6:
        return 2

    # Short entry conditions
    c1 = df['Low'].iloc[current_pos] < df['Open'].iloc[current_pos]
    c2 = df['Open'].iloc[current_pos] < df['Low'].iloc[current_pos - 2]
    c3 = df['Low'].iloc[current_pos - 2] < df['Low'].iloc[current_pos - 1]
    c4 = df['Low'].iloc[current_pos - 1] < df['High'].iloc[current_pos]
    c5 = df['High'].iloc[current_pos] < df['High'].iloc[current_pos - 2]
    c6 = df['High'].iloc[current_pos - 2] < df['High'].iloc[current_pos - 1]

    if c1 and c2 and c3 and c4 and c5 and c6:
        return 1

    return 0

# Function to add TotalSignal column
def add_total_signal(df):
    df['TotalSignal'] = df.apply(lambda row: total_signal(df, row.name), axis=1)
    return df

# Function to calculate Entry, SL, TP, Exit Point, and Profit/Loss
def add_entry_sl_tp_exit(df, tp_multiplier, sl_candle):
    df['Entry'] = np.nan
    df['SL'] = np.nan
    df['TP'] = np.nan
    df['ExitPoint'] = np.nan
    df['ProfitLoss'] = np.nan

    for i in range(2, len(df) - 1):
        if df['TotalSignal'].iloc[i] == 2:  # Long Signal
            entry = df['Open'].iloc[i + 1]
            sl = df['Low'].iloc[i] if sl_candle == 'Current Candle' else df['Low'].iloc[i - 1]
            tp = entry + tp_multiplier * (entry - sl)

            df.at[df.index[i], 'Entry'] = entry
            df.at[df.index[i], 'SL'] = sl
            df.at[df.index[i], 'TP'] = tp

            # Evaluate subsequent candles for SL or TP hit
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= sl:
                    df.at[df.index[i], 'ExitPoint'] = sl
                    df.at[df.index[i], 'ProfitLoss'] = sl - entry
                    break
                elif df['High'].iloc[j] >= tp:
                    df.at[df.index[i], 'ExitPoint'] = tp
                    df.at[df.index[i], 'ProfitLoss'] = tp - entry
                    break

        elif df['TotalSignal'].iloc[i] == 1:  # Short Signal
            entry = df['Open'].iloc[i + 1]
            sl = df['High'].iloc[i] if sl_candle == 'Current Candle' else df['High'].iloc[i - 1]
            tp = entry - tp_multiplier * (sl - entry)

            df.at[df.index[i], 'Entry'] = entry
            df.at[df.index[i], 'SL'] = sl
            df.at[df.index[i], 'TP'] = tp

            # Evaluate subsequent candles for SL or TP hit
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= sl:
                    df.at[df.index[i], 'ExitPoint'] = sl
                    df.at[df.index[i], 'ProfitLoss'] = entry - sl
                    break
                elif df['Low'].iloc[j] <= tp:
                    df.at[df.index[i], 'ExitPoint'] = tp
                    df.at[df.index[i], 'ProfitLoss'] = entry - tp
                    break

    return df

# Function to calculate equity curve and max drawdown percentage
def calculate_equity_curve(df, starting_capital):
    df['CumulativeProfitLoss'] = df['ProfitLoss'].cumsum() + starting_capital
    df['Drawdown'] = df['CumulativeProfitLoss'] - df['CumulativeProfitLoss'].cummax()
    df['DrawdownPercentage'] = (df['Drawdown'] / df['CumulativeProfitLoss'].cummax()) * 100
    max_drawdown = df['DrawdownPercentage'].min()
    return df, max_drawdown

# Function to calculate key statistics
def calculate_statistics(df):
    total_profit_loss = df['ProfitLoss'].sum()
    win_trades = df[df['ProfitLoss'] > 0]['ProfitLoss'].count()
    loss_trades = df[df['ProfitLoss'] < 0]['ProfitLoss'].count()
    win_loss_ratio = win_trades / loss_trades if loss_trades != 0 else "N/A"
    avg_profit_per_trade = df['ProfitLoss'].mean()
    max_drawdown_percentage = df['DrawdownPercentage'].min()
    recovery_time = (df[df['CumulativeProfitLoss'].cummax() == df['CumulativeProfitLoss']].index[-1] - df['DrawdownPercentage'].idxmin()).days
    best_trade = df['ProfitLoss'].max()
    worst_trade = df['ProfitLoss'].min()
    total_trades = win_trades + loss_trades

    statistics = {
        "Total Profit/Loss": f"{total_profit_loss:.2f}",
        "Win/Loss Ratio": f"{win_loss_ratio:.2f}" if isinstance(win_loss_ratio, float) else win_loss_ratio,
        "Average Profit per Trade": f"{avg_profit_per_trade:.2f}",
        "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}",
        "Recovery Time (Days)": recovery_time,
        "Best Trade": f"{best_trade:.2f}",
        "Worst Trade": f"{worst_trade:.2f}",
        "Total Trades": total_trades
    }
    return statistics

# Function to plot cumulative profit graph
def plot_cumulative_profit(df):
    fig = go.Figure()

    # Plot cumulative profit curve
    fig.add_trace(go.Scatter(
        x=df.index, y=df['CumulativeProfitLoss'],
        mode='lines', name='Cumulative Profit'
    ))

    # Annotate max drawdown
    drawdown_idx = df['DrawdownPercentage'].idxmin()
    fig.add_trace(go.Scatter(
        x=[drawdown_idx],
        y=[df['CumulativeProfitLoss'].loc[drawdown_idx]],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=f'Max Drawdown: {df["DrawdownPercentage"].min():.2f}%',
        textposition='top center',
        name='Max Drawdown'
    ))

    # Recovery time annotation
    recovery_idx = df[df['CumulativeProfitLoss'].cummax() == df['CumulativeProfitLoss']].index[-1]
    recovery_time = (recovery_idx - drawdown_idx).days
    fig.add_trace(go.Scatter(
        x=[recovery_idx],
        y=[df['CumulativeProfitLoss'].loc[recovery_idx]],
        mode='markers+text',
        marker=dict(color='green', size=10),
        text=f'Recovery Time: {recovery_time} days',
        textposition='top center',
        name='Recovery Time'
    ))

    fig.update_layout(
        title="Cumulative Profit Curve with Max Drawdown & Recovery Time",
        xaxis_title="Date", yaxis_title="Cumulative Profit (Rs)",
        legend_title="Legend"
    )
    return fig

# Streamlit App with Downloadable Report and Advanced Features
def main():
    st.title("Enhanced Backtesting Analysis")

    # File upload and inputs
   # Streamlit App with Downloadable Report and Advanced Features (continued)
    uploaded_file = st.file_uploader("Upload Historical Data CSV", type="csv")
    tp_multiplier = st.selectbox("Select TP Factor", [1, 1.5, 2, 2.5, 3])
    sl_candle = st.radio("Select SL Candle", ["Current Candle", "Previous Candle"])
    starting_capital = st.number_input("Enter Starting Capital (Rs)", value=1000000, step=10000)

    if uploaded_file:
        # Load and process CSV
        df = read_csv_to_dataframe(uploaded_file)
        df = add_total_signal(df)
        df = add_entry_sl_tp_exit(df, tp_multiplier, sl_candle)

        # Calculate equity curve and max drawdown
        df, max_drawdown = calculate_equity_curve(df, starting_capital)

        # Show processed data and max drawdown
        st.write("Processed Data", df)
        st.write(f"Max Drawdown: {max_drawdown:.2f}%")

        # Generate Cumulative Profit graph
        st.plotly_chart(plot_cumulative_profit(df))

        # Generate and display key statistics
        stats = calculate_statistics(df)
        st.write("Backtesting Key Statistics")
        for key, value in stats.items():
            st.write(f"{key}: {value}")

        # Generate Year-Month summary with Grand Total and Conditional Formatting
        st.write("Year-Month Summary")
        year_month_summary = df.copy()
        year_month_summary['Year'] = year_month_summary.index.year
        year_month_summary['Month'] = year_month_summary.index.month
        summary = year_month_summary.groupby(['Year', 'Month'])['ProfitLoss'].sum().unstack(fill_value=0)

        # Add cumulative profits column starting from 0
        cumulative_profits = summary.cumsum(axis=1)
        summary['Grand Total'] = summary.sum(axis=1)  # Add Grand Total column

        # Apply conditional formatting for positive/negative values
        formatted_summary = summary.style.applymap(
            lambda x: 'background-color: green;' if x > 0 else 'background-color: red;', subset=summary.columns
        ).format(precision=2)

        st.dataframe(formatted_summary)

if __name__ == "__main__":
    main()
