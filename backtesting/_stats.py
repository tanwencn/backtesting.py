from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from ._util import _data_period

if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        equity_real: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: 'Strategy',
        risk_free_rate: float = 0,
        attachment: bool = True,
) -> pd.Series:
    assert -1 < risk_free_rate < 1
    cumprod_close = ((1 + ohlc_data.Close.pct_change()).cumprod() - 1).to_numpy() *100
    index = ohlc_data.index
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    # 计算超额收益
    overage_equity = (equity_real / equity_real[0]-1) * 100-cumprod_close

    equity_df = pd.DataFrame({
        'Equity Overage': overage_equity,
        'Equity Hold Buy': cumprod_close,
        'Equity': equity,
        'Equity Real': equity_real,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)

    if isinstance(trades, pd.DataFrame):
        trades_df: pd.DataFrame = trades
    else:
        # Came straight from Backtest.run()
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'PnL': [t.pl for t in trades],
            'ReturnPct': [t.pl_pct for t in trades],
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
            'Tag': [t.tag for t in trades],
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
        trades_df['DurationBar'] = trades_df['ExitBar'] - trades_df['EntryBar']
    del trades

    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']
    durationsBar = trades_df['DurationBar']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series(dtype=object)
    s.loc['开始时间'] = index[0]
    s.loc['结束时间'] = index[-1]
    s.loc['Duration'] = s.结束时间 - s.开始时间

    if len(trades_df['EntryTime']) > 0 :
        # 将交易按年份分组，并计算每年的盈利情况
        profits = trades_df.groupby(trades_df['EntryTime'].dt.year)['PnL'].sum()
        # 统计盈利年份数
        s.loc['盈亏年数[%]'] = round((profits > 0).sum()/len(profits), 2)*100
    else:
        s.loc['盈亏年数[%]'] = 0.0

    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1

    #s.loc['Exposure Time [%]'] = have_position.mean() * 100  # In "n bars" time, not index time
    # s.loc['Equity Final [$]'] = equity[-1]
    # s.loc['Equity Peak [$]'] = equity.max()
    #s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    s.loc['市场参与度[%]'] = have_position.mean() * 100  # In "n bars" time, not index time
    s.loc['最终净值[$]'] = equity[-1]
    s.loc['最高净值[$]'] = equity.max()
    s.loc['总收益率[%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    #s.loc['Buy & Hold Return [%]'] = (c[-1] - c[0]) / c[0] * 100  # long-only return
    s.loc['买入并持有[%]'] = equity_df['Equity Hold Buy'][-1]  # long-only return
    s.loc['超额收益[%]'] = equity_df['Equity Overage'][-1]

    gmean_day_return: float = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    if isinstance(index, pd.DatetimeIndex):
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)
        annual_trading_days = float(
            365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else
            252)

    # Annualized return and risk metrics are computed based on the (mostly correct)
    # assumption that the returns are compounded. See: https://dx.doi.org/10.2139/ssrn.3054517
    # Our annualized return matches `empyrical.annual_return(day_returns)` whereas
    # our risk doesn't; they use the simpler approach below.
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    # s.loc['Return (Ann.) [%]'] = annualized_return * 100
    # s.loc['Volatility (Ann.) [%]'] = np.sqrt(
    #     (day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return) ** 2) ** annual_trading_days - (
    #                 1 + gmean_day_return) ** (2 * annual_trading_days)) * 100  # noqa: E501
    s.loc['年化收益率[%]'] = annualized_return * 100
    s.loc['年化波动率[%]'] = np.sqrt(
        (day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return) ** 2) ** annual_trading_days - (
                    1 + gmean_day_return) ** (2 * annual_trading_days)) * 100  # noqa: E501
    # s.loc['Return (Ann.) [%]'] = gmean_day_return * annual_trading_days * 100
    # s.loc['Risk (Ann.) [%]'] = day_returns.std(ddof=1) * np.sqrt(annual_trading_days) * 100

    # Our Sharpe mismatches `empyrical.sharpe_ratio()` because they use arithmetic mean return
    # and simple standard deviation
    #s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate) / (s.loc['Volatility (Ann.) [%]'] or np.nan)  # noqa: E501
    s.loc['夏普比率'] = (s.loc['年化收益率[%]'] - risk_free_rate) / (s.loc['年化波动率[%]'] or np.nan)  # noqa: E501
    # Our Sortino mismatches `empyrical.sortino_ratio()` because they use arithmetic mean return
    max_dd = -np.nan_to_num(dd.max())

    s.loc['最大回撤率[%]'] = max_dd * 100
    s.loc['平均回撤率[%]'] = -dd_peaks.mean() * 100
    s.loc['最大回撤周期'] = _round_timedelta(dd_dur.max())
    s.loc['平均回撤周期'] = _round_timedelta(dd_dur.mean())
    s.loc['交易次数'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['胜率[%]'] = win_rate * 100
    s.loc['最佳交易率[%]'] = returns.max() * 100
    s.loc['最差交易率[%]'] = returns.min() * 100
    mean_return = geometric_mean(returns)
    s.loc['每笔交易平均收益率[%]'] = mean_return * 100
    #s.loc['最大持仓周期'] = _round_timedelta(durations.max())
    #s.loc['平均持仓周期'] = _round_timedelta(durations.mean())
    s.loc['最大持仓周期'] = durationsBar.max()
    s.loc['平均持仓周期'] = durationsBar.mean()
    s.loc['盈利因子'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)  # noqa: E501
    s.loc['期望收益率[%]'] = returns.mean() * 100
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    #s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())
    #s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(day_returns.clip(-np.inf, 0) ** 2)) * np.sqrt(annual_trading_days))  # noqa: E501
    #s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)

    s.loc['_strategy'] = strategy_instance
    if attachment:
        s.loc['_equity_curve'] = equity_df
        s.loc['_trades'] = trades_df

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    def __repr__(self):
        # Prevent expansion due to _equity and _trades dfs
        with pd.option_context('max_colwidth', 20):
            return super().__repr__()
