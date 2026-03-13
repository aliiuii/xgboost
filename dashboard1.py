"""
XGBoost Trading - Live Web Dashboard
=====================================
Versi Stabil: Perbaikan Sinkronisasi Data & JSON Safety.
Design & UI: 100% Original sesuai permintaan.
"""

import os, sys, json, time, threading
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M15
TIMEFRAME_NAME = "M15"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_xgb_model_v5.json")

SL_ATR_MULT = 2.5
RR_RATIO = 2.0
CONFIDENCE_THRESHOLD = 0.55
MAX_SPREAD_POINTS = 50
MAGIC_NUMBER = 202603
DASHBOARD_BARS = 150
SIGNAL_LOOKBACK = 30
REFRESH_SECONDS = 30

PORT = 5555

# ============================================================
# PERSISTENCE
# ============================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORWARD_TEST_FILE = os.path.join(_BASE_DIR, "forward_test_history.json")
SIGNAL_HISTORY_FILE = os.path.join(_BASE_DIR, "signal_history.json")

def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except: pass
    return None

def _save_json(path, data):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except: pass

# --- Initialize Global Storage ---
forward_lock = threading.Lock()
_persisted = _load_json(FORWARD_TEST_FILE)
if _persisted and isinstance(_persisted, dict):
    forward_trades = _persisted.get('completed', [])
    forward_open  = _persisted.get('open', [])
else:
    forward_trades, forward_open = [], []

_sig_persisted = _load_json(SIGNAL_HISTORY_FILE)
signal_history = _sig_persisted if isinstance(_sig_persisted, list) else []
signal_history_lock = threading.Lock()

# Live Trading State
live_trade_enabled = False
live_trade_lock = threading.Lock()
last_trade_bar_time = None
LOT_SIZE = 0.01

def execute_trade(signal_data, bar_time):
    global last_trade_bar_time
    with live_trade_lock:
        if last_trade_bar_time == bar_time:
            return  # Already traded this bar
        
        # Check current positions to avoid multiple positions (optional, here we limit to 1 overall or just 1 per bar)
        open_pos = mt5.positions_get(symbol=SYMBOL)
        if open_pos is not None and len([p for p in open_pos if p.magic == MAGIC_NUMBER]) >= 1:
            return # Max open trades reached (adjust if you want more)

        order_type = mt5.ORDER_TYPE_BUY if signal_data['signal'] == 'BUY' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": float(signal_data['sl']),
            "tp": float(signal_data['tp']),
            "deviation": MAX_SPREAD_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": "XGB Live Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            last_trade_bar_time = bar_time
            print(f"Trade executed: {signal_data['signal']} at {price}")
        else:
            print(f"Trade failed: {result.retcode} - {result.comment}")

def get_mt5_history_summary():
    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None or len(deals) == 0:
        return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'profit': 0.0, 'history': []}
    
    history = []
    wins = 0
    losses = 0
    total_profit = 0.0
    
    for d in deals:
        if d.symbol == SYMBOL and d.magic == MAGIC_NUMBER and d.entry == 1: # DEAL_ENTRY_OUT
            is_win = d.profit > 0
            if is_win: wins += 1
            elif d.profit < 0: losses += 1
            
            total_profit += d.profit
            history.append({
                'time': pd.to_datetime(d.time, unit='s').strftime('%m/%d %H:%M'),
                'type': 'BUY' if d.type == 1 else 'SELL',
                'volume': float(d.volume),
                'profit': float(d.profit)
            })
            
    total = wins + losses
    wr = round(wins / total * 100, 1) if total > 0 else 0.0
    
    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'win_rate': wr,
        'profit': round(total_profit, 2),
        'history': history[-20:] # Last 20 trades
    }

# ============================================================
# FEATURE ENGINEERING FUNCTIONS (ASLI)
# ============================================================

def download_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df

def download_h1_data(symbol, num_m15_bars):
    h1_bars = num_m15_bars // 4 + 200
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, h1_bars)
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df

def find_swing_points(df, left=5, right=5):
    highs = df['high'].values; lows = df['low'].values; n = len(df)
    swing_highs = np.full(n, np.nan); swing_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        if highs[i] == max(highs[i-left:i+right+1]): swing_highs[i] = highs[i]
        if lows[i] == min(lows[i-left:i+right+1]): swing_lows[i] = lows[i]
    return swing_highs, swing_lows

def get_sr_levels(swing_highs, swing_lows, current_idx, max_levels=10):
    sh = swing_highs[:current_idx]; sl = swing_lows[:current_idx]
    resistance_levels = sh[~np.isnan(sh)][-max_levels:] if np.any(~np.isnan(sh)) else np.array([])
    support_levels = sl[~np.isnan(sl)][-max_levels:] if np.any(~np.isnan(sl)) else np.array([])
    return support_levels, resistance_levels

def compute_sr_features(df, swing_highs, swing_lows, max_levels=10):
    n = len(df); close = df['close'].values; high = df['high'].values; low = df['low'].values
    dist_nearest_support = np.full(n, 0.0); dist_nearest_resistance = np.full(n, 0.0)
    num_supports_below = np.full(n, 0.0); num_resistances_above = np.full(n, 0.0)
    sr_zone_strength = np.full(n, 0.0); price_position_in_sr = np.full(n, 0.5)
    retest_support = np.full(n, 0.0); retest_resistance = np.full(n, 0.0)
    bounce_from_support = np.full(n, 0.0); rejection_from_resistance = np.full(n, 0.0)
    for i in range(20, n):
        supports, resistances = get_sr_levels(swing_highs, swing_lows, i, max_levels)
        price = close[i]
        if len(supports) == 0 and len(resistances) == 0: continue
        s_below = supports[supports < price] if len(supports) > 0 else np.array([])
        r_above = resistances[resistances > price] if len(resistances) > 0 else np.array([])
        num_supports_below[i] = len(s_below); num_resistances_above[i] = len(r_above)
        atr = np.mean(high[max(0,i-14):i] - low[max(0,i-14):i]) if i > 14 else 1.0
        if atr == 0: atr = 1.0
        if len(s_below) > 0:
            nearest_s = s_below[-1]; dist_nearest_support[i] = (price - nearest_s) / atr
            if dist_nearest_support[i] < 0.5:
                retest_support[i] = 1.0
                if low[i] <= nearest_s * 1.001 and close[i] > nearest_s: bounce_from_support[i] = 1.0
        if len(r_above) > 0:
            nearest_r = r_above[0]; dist_nearest_resistance[i] = (nearest_r - price) / atr
            if dist_nearest_resistance[i] < 0.5:
                retest_resistance[i] = 1.0
                if high[i] >= nearest_r * 0.999 and close[i] < nearest_r: rejection_from_resistance[i] = 1.0
        if len(s_below) > 0 and len(r_above) > 0:
            sr_range = r_above[0] - s_below[-1]
            if sr_range > 0: price_position_in_sr[i] = (price - s_below[-1]) / sr_range
        zone_threshold = atr * 0.3; all_sr = np.concatenate([supports, resistances])
        touches = np.sum(np.abs(all_sr - price) < zone_threshold); sr_zone_strength[i] = touches
    return {'dist_nearest_support': dist_nearest_support, 'dist_nearest_resistance': dist_nearest_resistance,
            'num_supports_below': num_supports_below, 'num_resistances_above': num_resistances_above,
            'sr_zone_strength': sr_zone_strength, 'price_position_in_sr': price_position_in_sr,
            'retest_support': retest_support, 'retest_resistance': retest_resistance,
            'bounce_from_support': bounce_from_support, 'rejection_from_resistance': rejection_from_resistance}

def compute_trendline_features(df, swing_highs, swing_lows, min_touches=2, lookback_bars=100):
    n = len(df); close = df['close'].values; high = df['high'].values; low = df['low'].values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]; atr = pd.Series(tr).rolling(14).mean().values
    dist_up_trendline = np.zeros(n); dist_down_trendline = np.zeros(n)
    up_trendline_slope = np.zeros(n); down_trendline_slope = np.zeros(n)
    price_above_up_tl = np.zeros(n); price_below_down_tl = np.zeros(n)
    up_tl_touch = np.zeros(n); down_tl_touch = np.zeros(n)
    up_tl_break = np.zeros(n); down_tl_break = np.zeros(n); tl_squeeze = np.zeros(n)
    def fit_trendline(points_x, points_y):
        if len(points_x) < min_touches: return None
        x = np.array(points_x, dtype=np.float64); y = np.array(points_y, dtype=np.float64)
        n_pts = len(x); sum_x = np.sum(x); sum_y = np.sum(y)
        sum_xx = np.sum(x * x); sum_xy = np.sum(x * y)
        denom = n_pts * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10: return None
        slope = (n_pts * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n_pts
        return slope, intercept
    for i in range(lookback_bars, n):
        curr_atr = atr[i]
        if np.isnan(curr_atr) or curr_atr <= 0: curr_atr = 1.0
        sl_indices = []; sl_values = []
        for j in range(max(0, i - lookback_bars), i):
            if not np.isnan(swing_lows[j]): sl_indices.append(j); sl_values.append(swing_lows[j])
        sh_indices = []; sh_values = []
        for j in range(max(0, i - lookback_bars), i):
            if not np.isnan(swing_highs[j]): sh_indices.append(j); sh_values.append(swing_highs[j])
        if len(sl_indices) >= min_touches:
            use_n = min(5, len(sl_indices)); sl_x = sl_indices[-use_n:]; sl_y = sl_values[-use_n:]
            result = fit_trendline(sl_x, sl_y)
            if result is not None:
                slope, intercept = result; tl_value_at_i = slope * i + intercept
                if slope > 0:
                    dist = (close[i] - tl_value_at_i) / curr_atr
                    dist_up_trendline[i] = np.clip(dist, -10, 10)
                    up_trendline_slope[i] = np.clip(slope / curr_atr * 10, -10, 10)
                    price_above_up_tl[i] = 1.0 if close[i] > tl_value_at_i else 0.0
                    if abs(dist) < 0.3: up_tl_touch[i] = 1.0
                    if close[i] < tl_value_at_i and i > 0:
                        prev_tl = slope * (i-1) + intercept
                        if close[i-1] > prev_tl: up_tl_break[i] = 1.0
        if len(sh_indices) >= min_touches:
            use_n = min(5, len(sh_indices)); sh_x = sh_indices[-use_n:]; sh_y = sh_values[-use_n:]
            result = fit_trendline(sh_x, sh_y)
            if result is not None:
                slope, intercept = result; tl_value_at_i = slope * i + intercept
                if slope < 0:
                    dist = (tl_value_at_i - close[i]) / curr_atr
                    dist_down_trendline[i] = np.clip(dist, -10, 10)
                    down_trendline_slope[i] = np.clip(slope / curr_atr * 10, -10, 10)
                    price_below_down_tl[i] = 1.0 if close[i] < tl_value_at_i else 0.0
                    if abs(dist) < 0.3: down_tl_touch[i] = 1.0
                    if close[i] > tl_value_at_i and i > 0:
                        prev_tl = slope * (i-1) + intercept
                        if close[i-1] < prev_tl: down_tl_break[i] = 1.0
        if dist_up_trendline[i] != 0 and dist_down_trendline[i] != 0:
            total_dist = abs(dist_up_trendline[i]) + abs(dist_down_trendline[i])
            if total_dist < 2.0: tl_squeeze[i] = 1.0
    return {'dist_up_trendline': dist_up_trendline, 'dist_down_trendline': dist_down_trendline,
            'up_trendline_slope': up_trendline_slope, 'down_trendline_slope': down_trendline_slope,
            'price_above_up_tl': price_above_up_tl, 'price_below_down_tl': price_below_down_tl,
            'up_tl_touch': up_tl_touch, 'down_tl_touch': down_tl_touch,
            'up_tl_break': up_tl_break, 'down_tl_break': down_tl_break, 'tl_squeeze': tl_squeeze}

def compute_market_structure(df):
    close = df['close'].values; high = df['high'].values; low = df['low'].values; n = len(df)
    ema_fast = pd.Series(close).ewm(span=8).mean().values
    ema_mid = pd.Series(close).ewm(span=21).mean().values
    ema_slow = pd.Series(close).ewm(span=50).mean().values
    features = {}
    features['ema_fast_mid_diff'] = (ema_fast - ema_mid) / ema_mid * 100
    features['ema_mid_slow_diff'] = (ema_mid - ema_slow) / ema_slow * 100
    features['price_above_ema_fast'] = (close > ema_fast).astype(float)
    features['price_above_ema_mid'] = (close > ema_mid).astype(float)
    features['price_above_ema_slow'] = (close > ema_slow).astype(float)
    window = 10; hh = np.zeros(n); ll = np.zeros(n)
    for i in range(window*2, n):
        prev_high = np.max(high[i-window*2:i-window]); curr_high = np.max(high[i-window:i])
        prev_low = np.min(low[i-window*2:i-window]); curr_low = np.min(low[i-window:i])
        hh[i] = 1.0 if curr_high > prev_high else 0.0
        ll[i] = 1.0 if curr_low < prev_low else 0.0
    features['higher_high'] = hh; features['lower_low'] = ll; features['trend_score'] = hh - ll
    return features

def compute_technical_indicators(df):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    volume = df['volume'].values.astype(float); n = len(df); features = {}
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]; atr = pd.Series(tr).rolling(14).mean().values
    features['atr_normalized'] = atr / close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values; avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100); features['rsi'] = 100 - (100 / (1 + rs))
    low_14 = pd.Series(low).rolling(14).min().values; high_14 = pd.Series(high).rolling(14).max().values
    denom = high_14 - low_14; denom = np.where(denom == 0, 1, denom)
    features['stoch_k'] = ((close - low_14) / denom) * 100
    features['stoch_d'] = pd.Series(features['stoch_k']).rolling(3).mean().values
    ema12 = pd.Series(close).ewm(span=12).mean().values; ema26 = pd.Series(close).ewm(span=26).mean().values
    macd = ema12 - ema26; signal = pd.Series(macd).ewm(span=9).mean().values
    features['macd_normalized'] = macd / close * 1000
    features['macd_signal_diff'] = (macd - signal) / close * 1000
    features['macd_histogram'] = features['macd_signal_diff']
    sma20 = pd.Series(close).rolling(20).mean().values; std20 = pd.Series(close).rolling(20).std().values
    upper_bb = sma20 + 2 * std20; lower_bb = sma20 - 2 * std20
    features['bb_position'] = np.where(upper_bb - lower_bb != 0, (close - lower_bb) / (upper_bb - lower_bb + 1e-10), 0.5)
    features['bb_width'] = np.where(sma20 != 0, (upper_bb - lower_bb) / sma20, 0)
    vol_sma = pd.Series(volume).rolling(20).mean().values
    features['volume_ratio'] = np.where(vol_sma != 0, volume / (vol_sma + 1e-10), 1.0)
    body = close - df['open'].values; candle_range = high - low
    features['body_ratio'] = np.where(candle_range != 0, body / (candle_range + 1e-10), 0)
    features['upper_shadow'] = np.where(candle_range != 0, (high - np.maximum(close, df['open'].values)) / (candle_range + 1e-10), 0)
    features['lower_shadow'] = np.where(candle_range != 0, (np.minimum(close, df['open'].values) - low) / (candle_range + 1e-10), 0)
    features['return_1'] = np.concatenate([[0], np.diff(close) / close[:-1] * 100])
    features['return_3'] = pd.Series(close).pct_change(3).values * 100
    features['return_5'] = pd.Series(close).pct_change(5).values * 100
    features['return_10'] = pd.Series(close).pct_change(10).values * 100
    features['volatility_10'] = pd.Series(features['return_1']).rolling(10).std().values
    features['volatility_20'] = pd.Series(features['return_1']).rolling(20).std().values
    adx_period = 14; plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]; down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
    smoothed_tr = pd.Series(tr).ewm(span=adx_period, adjust=False).mean().values
    smoothed_plus_dm = pd.Series(plus_dm).ewm(span=adx_period, adjust=False).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(span=adx_period, adjust=False).mean().values
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    dx_denom = plus_di + minus_di
    dx = np.where(dx_denom != 0, 100 * np.abs(plus_di - minus_di) / dx_denom, 0)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False).mean().values
    features['adx'] = adx; features['plus_di'] = plus_di; features['minus_di'] = minus_di
    features['di_diff'] = plus_di - minus_di
    features['is_trending'] = (adx > 25).astype(float); features['is_ranging'] = (adx < 20).astype(float)
    hurst_window = 50; hurst = np.full(n, 0.5)
    for i in range(hurst_window, n):
        series = close[i-hurst_window:i]; mean_val = np.mean(series)
        deviations = series - mean_val; cumdev = np.cumsum(deviations)
        R = np.max(cumdev) - np.min(cumdev); S = np.std(series, ddof=1)
        if S > 0 and R > 0: hurst[i] = np.log(R / S) / np.log(hurst_window)
    features['hurst_exponent'] = hurst
    features['hurst_trending'] = (hurst > 0.55).astype(float)
    features['hurst_reverting'] = (hurst < 0.45).astype(float)
    bullish_vol = np.where(close > df['open'].values, volume, 0)
    bearish_vol = np.where(close < df['open'].values, volume, 0)
    vol_delta = bullish_vol - bearish_vol
    features['volume_delta'] = vol_delta / (vol_sma + 1e-10)
    cvd = pd.Series(vol_delta).rolling(10).sum().values
    features['cvd_10'] = cvd / (pd.Series(volume).rolling(10).sum().values + 1e-10)
    cvd20 = pd.Series(vol_delta).rolling(20).sum().values
    features['cvd_20'] = cvd20 / (pd.Series(volume).rolling(20).sum().values + 1e-10)
    bull_vol_sum = pd.Series(bullish_vol).rolling(10).sum().values
    total_vol_sum = pd.Series(volume).rolling(10).sum().values
    features['vol_imbalance'] = np.where(total_vol_sum > 0, bull_vol_sum / total_vol_sum, 0.5)
    vol_z = (volume - vol_sma) / (pd.Series(volume).rolling(20).std().values + 1e-10)
    features['volume_climax'] = (np.abs(vol_z) > 2.0).astype(float)
    return features

def compute_smc_features(df, swing_highs, swing_lows):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    open_p = df['open'].values; n = len(df)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().values
    atr = np.where(np.isnan(atr) | (atr < 1e-10), 1.0, atr)
    features = {}
    bullish_ob_dist = np.zeros(n); bearish_ob_dist = np.zeros(n)
    in_bullish_ob = np.zeros(n); in_bearish_ob = np.zeros(n); ob_strength = np.zeros(n)
    impulse_threshold = 1.5; bullish_obs = []; bearish_obs = []
    for i in range(3, n):
        body_i = close[i] - open_p[i]; body_prev = close[i-1] - open_p[i-1]
        move = (close[i] - close[i-2]) / atr[i]
        if body_prev < 0 and body_i > 0 and move > impulse_threshold:
            bullish_obs.append((high[i-1], low[i-1], abs(move), i-1))
            if len(bullish_obs) > 20: bullish_obs.pop(0)
        if body_prev > 0 and body_i < 0 and move < -impulse_threshold:
            bearish_obs.append((high[i-1], low[i-1], abs(move), i-1))
            if len(bearish_obs) > 20: bearish_obs.pop(0)
        for ob_h, ob_l, strength, ob_idx in reversed(bullish_obs):
            dist = (close[i] - (ob_h + ob_l) / 2) / atr[i]
            if abs(dist) < 10:
                bullish_ob_dist[i] = dist; ob_strength[i] = strength
                if close[i] >= ob_l and close[i] <= ob_h: in_bullish_ob[i] = 1.0
                break
        for ob_h, ob_l, strength, ob_idx in reversed(bearish_obs):
            dist = ((ob_h + ob_l) / 2 - close[i]) / atr[i]
            if abs(dist) < 10:
                bearish_ob_dist[i] = dist
                if close[i] >= ob_l and close[i] <= ob_h: in_bearish_ob[i] = 1.0
                break
    features['bullish_ob_dist'] = bullish_ob_dist; features['bearish_ob_dist'] = bearish_ob_dist
    features['in_bullish_ob'] = in_bullish_ob; features['in_bearish_ob'] = in_bearish_ob
    features['ob_strength'] = ob_strength
    bullish_fvg_dist = np.zeros(n); bearish_fvg_dist = np.zeros(n)
    in_bullish_fvg = np.zeros(n); in_bearish_fvg = np.zeros(n); fvg_size = np.zeros(n)
    bullish_fvgs = []; bearish_fvgs = []
    for i in range(2, n):
        if low[i] > high[i-2]:
            gap = (low[i] - high[i-2]) / atr[i]
            if gap > 0.1: bullish_fvgs.append((low[i], high[i-2], gap, i))
            if len(bullish_fvgs) > 15: bullish_fvgs.pop(0)
        if high[i] < low[i-2]:
            gap = (low[i-2] - high[i]) / atr[i]
            if gap > 0.1: bearish_fvgs.append((low[i-2], high[i], gap, i))
            if len(bearish_fvgs) > 15: bearish_fvgs.pop(0)
        for fvg_top, fvg_bot, size, fidx in reversed(bullish_fvgs):
            mid = (fvg_top + fvg_bot) / 2; dist = (close[i] - mid) / atr[i]
            if abs(dist) < 8:
                bullish_fvg_dist[i] = dist; fvg_size[i] = size
                if close[i] >= fvg_bot and close[i] <= fvg_top: in_bullish_fvg[i] = 1.0
                break
        for fvg_top, fvg_bot, size, fidx in reversed(bearish_fvgs):
            mid = (fvg_top + fvg_bot) / 2; dist = (mid - close[i]) / atr[i]
            if abs(dist) < 8:
                bearish_fvg_dist[i] = dist
                if close[i] >= fvg_bot and close[i] <= fvg_top: in_bearish_fvg[i] = 1.0
                break
    features['bullish_fvg_dist'] = bullish_fvg_dist; features['bearish_fvg_dist'] = bearish_fvg_dist
    features['in_bullish_fvg'] = in_bullish_fvg; features['in_bearish_fvg'] = in_bearish_fvg
    features['fvg_size'] = fvg_size
    bos_bullish = np.zeros(n); bos_bearish = np.zeros(n)
    choch_bullish = np.zeros(n); choch_bearish = np.zeros(n)
    structure_trend = np.zeros(n)
    recent_sh = []; recent_sl = []; current_trend = 0
    for i in range(10, n):
        if not np.isnan(swing_highs[i]): recent_sh.append((swing_highs[i], i))
        if len(recent_sh) > 10: recent_sh.pop(0)
        if not np.isnan(swing_lows[i]): recent_sl.append((swing_lows[i], i))
        if len(recent_sl) > 10: recent_sl.pop(0)
        if len(recent_sh) >= 2 and len(recent_sl) >= 2:
            last_sh = recent_sh[-1][0]; last_sl = recent_sl[-1][0]
            if close[i] > last_sh:
                if current_trend >= 0: bos_bullish[i] = 1.0
                else: choch_bullish[i] = 1.0
                current_trend = 1
            if close[i] < last_sl:
                if current_trend <= 0: bos_bearish[i] = 1.0
                else: choch_bearish[i] = 1.0
                current_trend = -1
        structure_trend[i] = current_trend
    features['bos_bullish'] = bos_bullish; features['bos_bearish'] = bos_bearish
    features['choch_bullish'] = choch_bullish; features['choch_bearish'] = choch_bearish
    features['structure_trend'] = structure_trend
    liq_sweep_high = np.zeros(n); liq_sweep_low = np.zeros(n)
    liq_grab_bullish = np.zeros(n); liq_grab_bearish = np.zeros(n); lookback_liq = 20
    for i in range(lookback_liq + 1, n):
        prev_high = np.max(high[i-lookback_liq:i]); prev_low = np.min(low[i-lookback_liq:i])
        if high[i] > prev_high and close[i] < prev_high:
            liq_sweep_high[i] = 1.0
            if close[i] < open_p[i]: liq_grab_bearish[i] = 1.0
        if low[i] < prev_low and close[i] > prev_low:
            liq_sweep_low[i] = 1.0
            if close[i] > open_p[i]: liq_grab_bullish[i] = 1.0
    features['liq_sweep_high'] = liq_sweep_high; features['liq_sweep_low'] = liq_sweep_low
    features['liq_grab_bullish'] = liq_grab_bullish; features['liq_grab_bearish'] = liq_grab_bearish
    premium_discount = np.zeros(n); in_premium = np.zeros(n); in_discount = np.zeros(n); range_window = 50
    for i in range(range_window, n):
        range_high = np.max(high[i-range_window:i]); range_low = np.min(low[i-range_window:i])
        range_size = range_high - range_low
        if range_size > 0:
            position = (close[i] - range_low) / range_size
            premium_discount[i] = position * 2 - 1
            if position > 0.618: in_premium[i] = 1.0
            elif position < 0.382: in_discount[i] = 1.0
    features['premium_discount'] = premium_discount; features['in_premium'] = in_premium; features['in_discount'] = in_discount
    displacement_up = np.zeros(n); displacement_down = np.zeros(n); displacement_strength = np.zeros(n)
    for i in range(1, n):
        body = close[i] - open_p[i]; body_atr = abs(body) / atr[i]
        if body_atr > 1.5:
            displacement_strength[i] = body_atr
            if body > 0: displacement_up[i] = 1.0
            else: displacement_down[i] = 1.0
    features['displacement_up'] = displacement_up; features['displacement_down'] = displacement_down
    features['displacement_strength'] = displacement_strength
    smc_bullish_score = np.zeros(n); smc_bearish_score = np.zeros(n)
    for i in range(1, n):
        bull = bear = 0
        if in_bullish_ob[i]: bull += 2
        if in_bearish_ob[i]: bear += 2
        if in_bullish_fvg[i]: bull += 1.5
        if in_bearish_fvg[i]: bear += 1.5
        if bos_bullish[i]: bull += 1
        if bos_bearish[i]: bear += 1
        if choch_bullish[i]: bull += 2
        if choch_bearish[i]: bear += 2
        if liq_grab_bullish[i]: bull += 2
        if liq_grab_bearish[i]: bear += 2
        if in_discount[i]: bull += 1
        if in_premium[i]: bear += 1
        if displacement_up[i]: bull += 1
        if displacement_down[i]: bear += 1
        smc_bullish_score[i] = bull; smc_bearish_score[i] = bear
    features['smc_bullish_score'] = smc_bullish_score; features['smc_bearish_score'] = smc_bearish_score
    features['smc_net_score'] = smc_bullish_score - smc_bearish_score
    return features

def compute_h1_context_features(df_m15, df_h1_raw):
    n = len(df_m15); features = {}
    if df_h1_raw is None or len(df_h1_raw) == 0:
        for name in ['h1_ema_trend','h1_rsi','h1_atr_ratio','h1_body_ratio','h1_trend_score','h1_bb_position','h1_volume_ratio']:
            features[name] = np.zeros(n)
        return features
    h1_close = df_h1_raw['close'].values; h1_high = df_h1_raw['high'].values
    h1_low = df_h1_raw['low'].values; h1_open = df_h1_raw['open'].values
    h1_vol = df_h1_raw['volume'].values.astype(float); nh1 = len(df_h1_raw)
    h1_ema8 = pd.Series(h1_close).ewm(span=8).mean().values
    h1_ema21 = pd.Series(h1_close).ewm(span=21).mean().values
    h1_delta = np.diff(h1_close, prepend=h1_close[0])
    h1_gain = np.where(h1_delta > 0, h1_delta, 0); h1_loss = np.where(h1_delta < 0, -h1_delta, 0)
    h1_avg_gain = pd.Series(h1_gain).rolling(14).mean().values
    h1_avg_loss = pd.Series(h1_loss).rolling(14).mean().values
    h1_rs = np.where(h1_avg_loss != 0, h1_avg_gain / h1_avg_loss, 100)
    h1_rsi = 100 - (100 / (1 + h1_rs))
    h1_tr = np.maximum(h1_high - h1_low, np.maximum(np.abs(h1_high - np.roll(h1_close, 1)), np.abs(h1_low - np.roll(h1_close, 1))))
    h1_tr[0] = h1_high[0] - h1_low[0]; h1_atr = pd.Series(h1_tr).rolling(14).mean().values
    h1_sma20 = pd.Series(h1_close).rolling(20).mean().values; h1_std20 = pd.Series(h1_close).rolling(20).std().values
    h1_upper_bb = h1_sma20 + 2 * h1_std20; h1_lower_bb = h1_sma20 - 2 * h1_std20
    h1_bb_pos = np.where(h1_upper_bb - h1_lower_bb != 0, (h1_close - h1_lower_bb) / (h1_upper_bb - h1_lower_bb + 1e-10), 0.5)
    h1_vol_sma = pd.Series(h1_vol).rolling(20).mean().values
    h1_times = df_h1_raw.index.values; m15_times = df_m15.index.values
    h1_idx_map = np.clip(np.searchsorted(h1_times, m15_times, side='right') - 1, 0, nh1 - 1)
    h1_ema_trend = np.zeros(n); h1_rsi_feat = np.zeros(n); h1_atr_ratio_feat = np.zeros(n)
    h1_body_ratio_feat = np.zeros(n); h1_trend_score_feat = np.zeros(n)
    h1_bb_pos_feat = np.zeros(n); h1_vol_ratio_feat = np.zeros(n)
    for i in range(n):
        hi = h1_idx_map[i]
        if hi < 50: continue
        h1_ema_trend[i] = (h1_ema8[hi] - h1_ema21[hi]) / (h1_ema21[hi] + 1e-10) * 100
        h1_rsi_feat[i] = h1_rsi[hi]
        h1_atr_ratio_feat[i] = h1_atr[hi] / (h1_close[hi] + 1e-10)
        h1_range = h1_high[hi] - h1_low[hi]; h1_body = h1_close[hi] - h1_open[hi]
        h1_body_ratio_feat[i] = h1_body / (h1_range + 1e-10) if h1_range > 0 else 0
        if hi >= 20:
            ph = np.max(h1_high[hi-20:hi-10]); ch = np.max(h1_high[hi-10:hi])
            pl = np.min(h1_low[hi-20:hi-10]); cl = np.min(h1_low[hi-10:hi])
            h1_trend_score_feat[i] = (1.0 if ch > ph else 0.0) - (1.0 if cl < pl else 0.0)
        h1_bb_pos_feat[i] = h1_bb_pos[hi]
        h1_vol_ratio_feat[i] = h1_vol[hi] / (h1_vol_sma[hi] + 1e-10) if h1_vol_sma[hi] > 0 else 1.0
    features['h1_ema_trend'] = h1_ema_trend; features['h1_rsi'] = h1_rsi_feat
    features['h1_atr_ratio'] = h1_atr_ratio_feat; features['h1_body_ratio'] = h1_body_ratio_feat
    features['h1_trend_score'] = h1_trend_score_feat; features['h1_bb_position'] = h1_bb_pos_feat
    features['h1_volume_ratio'] = h1_vol_ratio_feat
    return features

def build_features(df, df_h1=None):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    open_p = df['open'].values; n = len(df)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]; atr = pd.Series(tr).rolling(14).mean().values
    df['atr'] = atr
    swing_highs, swing_lows = find_swing_points(df, left=5, right=5)
    sr_f = compute_sr_features(df, swing_highs, swing_lows)
    tl_f = compute_trendline_features(df, swing_highs, swing_lows)
    ms_f = compute_market_structure(df)
    ti_f = compute_technical_indicators(df)
    smc_f = compute_smc_features(df, swing_highs, swing_lows)
    all_f = {}
    all_f.update(sr_f); all_f.update(tl_f); all_f.update(ms_f); all_f.update(ti_f)
    all_f['candle_body_pct'] = (close - open_p) / (atr + 1e-10)
    all_f['candle_range_pct'] = (high - low) / (atr + 1e-10)
    all_f.update(smc_f)
    hours = df.index.hour.values; dow = df.index.dayofweek.values
    all_f['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    all_f['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    all_f['day_sin'] = np.sin(2 * np.pi * dow / 5.0)
    all_f['day_cos'] = np.cos(2 * np.pi * dow / 5.0)
    all_f['session_asian'] = ((hours >= 0) & (hours < 8)).astype(float)
    all_f['session_london'] = ((hours >= 7) & (hours < 16)).astype(float)
    all_f['session_newyork'] = ((hours >= 13) & (hours < 22)).astype(float)
    all_f['session_overlap'] = ((hours >= 13) & (hours < 16)).astype(float)
    all_f['is_high_activity'] = (((hours >= 8) & (hours < 11)) | ((hours >= 13) & (hours < 16))).astype(float)
    h1_f = compute_h1_context_features(df, df_h1)
    all_f.update(h1_f)
    feature_df = pd.DataFrame(all_f, index=df.index)
    feature_df = feature_df.iloc[60:].copy()
    lag_cols = ['dist_nearest_support', 'dist_nearest_resistance', 'rsi', 'macd_normalized',
                'return_1', 'trend_score', 'bb_position', 'price_position_in_sr', 'candle_body_pct',
                'smc_net_score', 'structure_trend', 'premium_discount', 'bullish_ob_dist', 'bearish_ob_dist']
    for lag in [1, 2, 3, 5]:
        for col in lag_cols:
            if col in feature_df.columns:
                feature_df[f'{col}_lag{lag}'] = feature_df[col].shift(lag)
    change_cols = ['dist_nearest_support', 'rsi', 'macd_normalized', 'price_position_in_sr', 'smc_net_score', 'premium_discount']
    for col in change_cols:
        if col in feature_df.columns:
            feature_df[f'{col}_change'] = feature_df[col].diff()
    feature_df = feature_df.dropna()
    all_feature_cols = list(feature_df.columns)
    for col in all_feature_cols:
        feature_df[col] = feature_df[col].clip(-10, 10)
    X = feature_df[all_feature_cols].values.astype(np.float32)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return feature_df, all_feature_cols, scaler, X_scaled

# ============================================================
# FORWARD TEST TRACKER LOGIC
# ============================================================

def _save_forward_test():
    _save_json(FORWARD_TEST_FILE, {'completed': forward_trades, 'open': forward_open})

def _save_signal_history():
    with signal_history_lock:
        _save_json(SIGNAL_HISTORY_FILE, signal_history)

def append_signal_history(signals):
    with signal_history_lock:
        existing = {s['time'] for s in signal_history}
        added = 0
        for s in signals:
            if s['time'] not in existing:
                signal_history.append(s)
                existing.add(s['time'])
                added += 1
        if added > 0:
            _save_signal_history()

def check_forward_trades(candles_df):
    global forward_open, forward_trades
    if candles_df is None or len(candles_df) == 0: return
    with forward_lock:
        still_open = []; changed = False
        for trade in forward_open:
            resolved = False
            entry_time = pd.Timestamp(trade['entry_time'])
            future = candles_df[candles_df.index > entry_time]
            for t, row in future.iterrows():
                if trade['signal'] == 'BUY':
                    if row['low'] <= trade['sl']:
                        pts = int(round((trade['sl'] - trade['entry']) / POINT))
                        forward_trades.append({**trade, 'exit_time': str(t), 'exit_price': float(trade['sl']), 'result': 'LOSS', 'points': pts})
                        resolved = True; changed = True; break
                    if row['high'] >= trade['tp']:
                        pts = int(round((trade['tp'] - trade['entry']) / POINT))
                        forward_trades.append({**trade, 'exit_time': str(t), 'exit_price': float(trade['tp']), 'result': 'WIN', 'points': pts})
                        resolved = True; changed = True; break
                else:
                    if row['high'] >= trade['sl']:
                        pts = int(round((trade['entry'] - trade['sl']) / POINT))
                        forward_trades.append({**trade, 'exit_time': str(t), 'exit_price': float(trade['sl']), 'result': 'LOSS', 'points': pts})
                        resolved = True; changed = True; break
                    if row['low'] <= trade['tp']:
                        pts = int(round((trade['entry'] - trade['tp']) / POINT))
                        forward_trades.append({**trade, 'exit_time': str(t), 'exit_price': float(trade['tp']), 'result': 'WIN', 'points': pts})
                        resolved = True; changed = True; break
            if not resolved:
                if len(future) > 50:
                    last_close = float(future.iloc[-1]['close'])
                    pts = int(round((last_close - trade['entry']) / POINT) if trade['signal'] == 'BUY' else round((trade['entry'] - last_close) / POINT))
                    forward_trades.append({**trade, 'exit_time': str(future.index[-1]), 'exit_price': last_close, 'result': 'EXPIRED', 'points': pts})
                    changed = True
                else:
                    still_open.append(trade)
        forward_open = still_open
        if changed: _save_forward_test()

def register_forward_signals(signals, candles_df):
    global forward_open
    with forward_lock:
        existing_times = {t['entry_time'] for t in forward_open}
        existing_times.update({t['entry_time'] for t in forward_trades})
        added = 0
        for s in signals:
            if s['time'] not in existing_times:
                forward_open.append({'entry_time': s['time'], 'signal': s['signal'], 'entry': float(s['entry']), 'sl': float(s['sl']), 'tp': float(s['tp']), 'confidence': float(s['confidence'])})
                added += 1
        if added > 0: _save_forward_test()

def get_forward_test_summary():
    with forward_lock:
        completed = list(forward_trades)
        pending = list(forward_open)
    total = len(completed)
    wins = sum(1 for t in completed if t['result'] == 'WIN')
    losses = sum(1 for t in completed if t['result'] == 'LOSS')
    expired = sum(1 for t in completed if t['result'] == 'EXPIRED')
    total_pts = sum(t['points'] for t in completed)
    wr = round(wins / total * 100, 1) if total > 0 else 0.0
    return {'total_trades': total, 'wins': wins, 'losses': losses, 'expired': expired, 'win_rate': wr, 'total_points': total_pts, 'pending': len(pending), 'trades': completed[-20:], 'open_trades': pending}

# ============================================================
# API DATA GENERATOR (FIXED JSON SAFETY)
# ============================================================

_last_good_data = {'data': None}

def get_dashboard_data():
    # Ensure MT5 is initialized for the current thread/session
    if not mt5.initialize():
        result = {
            'symbol': SYMBOL, 'timeframe': TIMEFRAME_NAME, 'digits': DIGITS, 'point': POINT,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'candles': None, 'signals': [], 'current_signal': {'signal': 'NONE', 'confidence': 0, 'entry': 0, 'sl': 0, 'tp': 0},
            'tick': {}, 'account': {}, 'positions': [], 'forward_test': get_forward_test_summary(),
            'mt5_history': {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'profit': 0.0, 'history': []},
            'live_trade_enabled': live_trade_enabled,
            'warning': None,
            'error': 'MT5 connection lost. Please restart MT5.'
        }
        return result

    # Base result — always returned even on partial failure
    result = {
        'symbol': SYMBOL, 'timeframe': TIMEFRAME_NAME, 'digits': DIGITS, 'point': POINT,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'candles': None, 'signals': [], 'current_signal': {'signal': 'NONE', 'confidence': 0, 'entry': 0, 'sl': 0, 'tp': 0},
        'tick': {}, 'account': {}, 'positions': [], 'forward_test': get_forward_test_summary(),
        'mt5_history': get_mt5_history_summary(),
        'live_trade_enabled': live_trade_enabled,
        'warning': None,
    }

    # --- 1) Download candle data ---
    try:
        df = download_data(SYMBOL, TIMEFRAME, 2500)
        df_h1 = download_h1_data(SYMBOL, 2500)
    except Exception as e:
        df, df_h1 = None, None
        result['warning'] = f'Data download error: {e}'

    if df is None or len(df) == 0:
        # Fallback: return cached data if available
        if _last_good_data['data'] is not None:
            cached = _last_good_data['data'].copy()
            cached['timestamp'] = result['timestamp']
            cached['warning'] = 'Market closed or data unavailable — showing last known data'
            cached['current_signal'] = {'signal': 'MARKET_CLOSED', 'confidence': 0, 'entry': 0, 'sl': 0, 'tp': 0}
            return cached
        result['error'] = 'No data available — market may be closed'
        return result

    # --- 2) Build chart candles (always succeeds if df exists) ---
    try:
        df_chart = df.iloc[-DASHBOARD_BARS:].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        result['candles'] = {
            'time': [t.isoformat() for t in df_chart.index],
            'open': df_chart['open'].round(DIGITS).tolist(),
            'high': df_chart['high'].round(DIGITS).tolist(),
            'low': df_chart['low'].round(DIGITS).tolist(),
            'close': df_chart['close'].round(DIGITS).tolist(),
            'volume': [float(v) for v in df_chart['volume'].tolist()],
        }
    except Exception as e:
        result['warning'] = f'Chart build error: {e}'

    # --- 3) Tick, account, positions (never crash the whole API) ---
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            result['tick'] = {'bid': round(tick.bid, DIGITS), 'ask': round(tick.ask, DIGITS), 'spread': round((tick.ask - tick.bid) / POINT)}
    except Exception:
        tick = None

    try:
        acct = mt5.account_info()
        if acct:
            result['account'] = {'balance': round(acct.balance, 2), 'equity': round(acct.equity, 2), 'floating': round(acct.equity - acct.balance, 2), 'login': acct.login, 'server': acct.server}
    except Exception:
        pass

    try:
        positions = mt5.positions_get(symbol=SYMBOL)
        result['positions'] = [{'ticket': p.ticket, 'type': 'BUY' if p.type == 0 else 'SELL', 'volume': p.volume, 'open_price': round(p.price_open, DIGITS), 'sl': round(p.sl, DIGITS), 'tp': round(p.tp, DIGITS), 'profit': round(p.profit, 2)} for p in (positions or []) if p.magic == MAGIC_NUMBER]
    except Exception:
        pass

    # --- 4) ML prediction (separate — chart survives if this fails) ---
    try:
        feat_df, feat_cols, scaler, X_scaled = build_features(df, df_h1)
        probs = model.predict_proba(X_scaled)
        signals = []
        for idx in range(-SIGNAL_LOOKBACK, 0):
            p = probs[idx]
            pred_cls = int(np.argmax(p))
            bar_time = feat_df.index[idx]
            bar_pos = df.index.get_loc(bar_time)
            atr_v = float(df['atr'].iloc[bar_pos])
            if np.isnan(atr_v) or atr_v < POINT * 10: continue
            if pred_cls == 2: continue
            prob_sell, prob_buy = float(p[0]), float(p[1])
            max_p = max(prob_sell, prob_buy)
            if max_p < CONFIDENCE_THRESHOLD: continue
            sig_type = 'BUY' if prob_buy > prob_sell else 'SELL'
            sl_d = atr_v * SL_ATR_MULT; tp_d = sl_d * RR_RATIO
            bar_close = float(df['close'].iloc[bar_pos])
            signals.append({
                'time': bar_time.isoformat(), 'signal': sig_type, 'confidence': float(round(max_p, 4)),
                'entry': float(round(bar_close, DIGITS)),
                'sl': float(round(bar_close - sl_d if sig_type == 'BUY' else bar_close + sl_d, DIGITS)),
                'tp': float(round(bar_close + tp_d if sig_type == 'BUY' else bar_close - tp_d, DIGITS)),
                'prob_buy': float(round(prob_buy, 3)), 'prob_sell': float(round(prob_sell, 3)), 'prob_hold': float(round(float(p[2]), 3)),
            })
        result['signals'] = signals

        last_p = probs[-1]
        last_cls = int(np.argmax(last_p))
        last_atr = float(df['atr'].iloc[-1])
        current_signal = {'signal': 'NONE', 'confidence': 0, 'entry': 0, 'sl': 0, 'tp': 0}

        if tick and not np.isnan(last_atr) and last_atr > POINT * 10:
            spread_pts = (tick.ask - tick.bid) / POINT
            if spread_pts <= MAX_SPREAD_POINTS and last_cls != 2:
                pb, ps = float(last_p[1]), float(last_p[0])
                mp = max(pb, ps)
                if mp >= CONFIDENCE_THRESHOLD:
                    sig = 'BUY' if pb > ps else 'SELL'
                    sl_d = last_atr * SL_ATR_MULT; tp_d = sl_d * RR_RATIO
                    entry = tick.ask if sig == 'BUY' else tick.bid
                    current_signal = {
                        'signal': sig, 'confidence': float(round(mp, 4)),
                        'entry': float(round(entry, DIGITS)),
                        'sl': float(round(entry - sl_d if sig == 'BUY' else entry + sl_d, DIGITS)),
                        'tp': float(round(entry + tp_d if sig == 'BUY' else entry - tp_d, DIGITS)),
                        'sl_pts': int(round(sl_d / POINT)), 'tp_pts': int(round(tp_d / POINT)),
                    }
                else: current_signal = {'signal': 'LOW_CONF', 'confidence': float(round(mp, 4)), 'entry': 0, 'sl': 0, 'tp': 0}
            elif last_cls == 2: current_signal = {'signal': 'HOLD', 'confidence': float(round(float(last_p[2]), 4)), 'entry': 0, 'sl': 0, 'tp': 0}
        result['current_signal'] = current_signal

        append_signal_history(signals)
        register_forward_signals(signals, df)
        check_forward_trades(df)

        if live_trade_enabled and current_signal.get('signal') in ['BUY', 'SELL']:
            execute_trade(current_signal, str(df.index[-1]))
            
    except Exception as e:
        result['warning'] = f'ML prediction error: {e}'

    # Cache successful result for fallback
    _last_good_data['data'] = result
    return result

# ============================================================
# FLASK APP + HTML TEMPLATE (EXACTLY SAME DESIGN)
# ============================================================

app = Flask(__name__)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XGB Trading Dashboard — {{ symbol }}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #e0e0e0; font-family: 'Inter', sans-serif; overflow-x: hidden; }

  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #2a2a4a;
  }
  .header h1 { font-size: 18px; color: #ffd700; font-weight: 700; }
  .header .meta { font-size: 12px; color: #888; font-family: 'JetBrains Mono', monospace; }
  .header .status-dot { width:8px; height:8px; border-radius:50%; background:#00ff88; display:inline-block; margin-right:6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .main { display: grid; grid-template-columns: 1fr 320px; gap: 0; height: calc(100vh - 52px); }

  .chart-area { display:flex; flex-direction:column; min-width:0; }
  #priceChart { flex: 1; min-height: 0; }
  #volumeChart { height: 100px; }

  .sidebar {
    background: #16213e; border-left: 1px solid #2a2a4a;
    overflow-y: auto; padding: 0;
  }
  .sidebar::-webkit-scrollbar { width: 4px; }
  .sidebar::-webkit-scrollbar-thumb { background: #4a4a6a; border-radius: 2px; }

  .panel { padding: 14px 16px; border-bottom: 1px solid #2a2a4a; }
  .panel-title { font-size: 11px; font-weight: 700; color: #4a90d9; text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 10px; }

  .signal-box { text-align: center; padding: 16px; border-radius: 8px; margin-bottom: 8px; }
  .signal-box.buy { background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05)); border: 1px solid #00ff8844; }
  .signal-box.sell { background: linear-gradient(135deg, rgba(255,51,102,0.15), rgba(255,51,102,0.05)); border: 1px solid #ff336644; }
  .signal-box.hold { background: rgba(136,136,136,0.1); border: 1px solid #88888844; }
  .signal-box.none { background: rgba(136,136,136,0.05); border: 1px solid #44444444; }
  .signal-label { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .signal-label.buy { color: #00ff88; }
  .signal-label.sell { color: #ff3366; }
  .signal-label.hold { color: #888; }
  .signal-label.none { color: #555; }
  .signal-conf { font-size: 12px; color: #aaa; margin-top: 4px; }

  .data-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px;
    font-family: 'JetBrains Mono', monospace; }
  .data-label { color: #888; }
  .data-value { color: #e0e0e0; font-weight: 600; }
  .data-value.green { color: #00ff88; }
  .data-value.red { color: #ff3366; }
  .data-value.gold { color: #ffd700; }
  .data-value.warn { color: #ffaa00; }

  .pos-card { background: #1a1a2e; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
    border-left: 3px solid; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
  .pos-card.buy { border-color: #00ff88; }
  .pos-card.sell { border-color: #ff3366; }
  .pos-header { display: flex; justify-content: space-between; margin-bottom: 3px; }
  .pos-type { font-weight: 700; }
  .pos-type.buy { color: #00ff88; }
  .pos-type.sell { color: #ff3366; }

  .signal-row { display: flex; align-items: center; padding: 5px 0; font-size: 11px;
    font-family: 'JetBrains Mono', monospace; border-bottom: 1px solid #1a1a2e; }
  .signal-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
  .signal-dot.buy { background: #00ff88; }
  .signal-dot.sell { background: #ff3366; }
  .signal-time { color: #888; margin-right: 8px; min-width: 90px; }
  .signal-conf-bar { height: 4px; border-radius: 2px; margin-left: auto; min-width: 40px; }

  .countdown { text-align: center; padding: 6px; font-size: 11px; color: #666;
    font-family: 'JetBrains Mono', monospace; background: #0d1117; }

  .live-trade-btn {
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #4a4a6a;
    padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.2s; font-family: 'Inter', sans-serif;
  }
  .live-trade-btn.active {
    background: rgba(0, 255, 136, 0.15); color: #00ff88; border-color: #00ff88;
  }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:12px;">
    <h1>&#x1F4CA; XGB Dashboard</h1>
    <span class="meta" style="color:#ffd700;font-weight:700;">{{ symbol }}</span>
    <span class="meta">{{ timeframe }}</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <button id="toggleTradeBtn" class="live-trade-btn" onclick="toggleLiveTrade()">Live Trade: OFF</button>
    <span class="status-dot"></span>
    <span class="meta" id="updateTime">--</span>
    <span class="meta" style="color:#4a90d9;">Auto-refresh {{ refresh }}s</span>
  </div>
</div>

<div class="main">
  <div class="chart-area">
    <div id="priceChart"></div>
    <div id="volumeChart"></div>
  </div>

  <div class="sidebar">
    <div class="panel">
      <div class="panel-title">Current Signal</div>
      <div id="signalBox" class="signal-box none">
        <div id="signalLabel" class="signal-label none">---</div>
        <div id="signalConf" class="signal-conf">Waiting...</div>
      </div>
      <div id="signalLevels" style="display:none;">
        <div class="data-row"><span class="data-label">Entry</span><span id="sigEntry" class="data-value gold">-</span></div>
        <div class="data-row"><span class="data-label">Stop Loss</span><span id="sigSL" class="data-value red">-</span></div>
        <div class="data-row"><span class="data-label">Take Profit</span><span id="sigTP" class="data-value green">-</span></div>
        <div class="data-row"><span class="data-label">SL dist</span><span id="sigSLPts" class="data-value">-</span></div>
        <div class="data-row"><span class="data-label">TP dist</span><span id="sigTPPts" class="data-value">-</span></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Price</div>
      <div class="data-row"><span class="data-label">Bid</span><span id="bid" class="data-value">-</span></div>
      <div class="data-row"><span class="data-label">Ask</span><span id="ask" class="data-value">-</span></div>
      <div class="data-row"><span class="data-label">Spread</span><span id="spread" class="data-value">-</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Account</div>
      <div class="data-row"><span class="data-label">Balance</span><span id="balance" class="data-value">-</span></div>
      <div class="data-row"><span class="data-label">Equity</span><span id="equity" class="data-value">-</span></div>
      <div class="data-row"><span class="data-label">Floating</span><span id="floating" class="data-value">-</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Bot Positions <span id="posCount" style="color:#e0e0e0;"></span></div>
      <div id="positionsContainer"><span style="color:#555;font-size:12px;">No positions</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">MT5 Live History</div>
      <div id="mt5Summary" style="margin-bottom:8px;">
        <div class="data-row"><span class="data-label">Total Trades</span><span id="mt5Total" class="data-value">0</span></div>
        <div class="data-row"><span class="data-label">Win / Loss</span><span id="mt5WL" class="data-value">0 / 0</span></div>
        <div class="data-row"><span class="data-label">Win Rate</span><span id="mt5WR" class="data-value">0%</span></div>
        <div class="data-row"><span class="data-label">Net Profit</span><span id="mt5Profit" class="data-value">$0.00</span></div>
      </div>
      <div id="mt5Trades" style="max-height:150px;overflow-y:auto;"></div>
    </div>

    <div class="panel">
      <div class="panel-title">Forward Test</div>
      <div id="fwdSummary" style="margin-bottom:8px;">
        <div class="data-row"><span class="data-label">Total Trades</span><span id="fwdTotal" class="data-value">0</span></div>
        <div class="data-row"><span class="data-label">Win / Loss / Exp</span><span id="fwdWL" class="data-value">0 / 0 / 0</span></div>
        <div class="data-row"><span class="data-label">Win Rate</span><span id="fwdWR" class="data-value">0%</span></div>
        <div class="data-row"><span class="data-label">Total Points</span><span id="fwdPts" class="data-value">0</span></div>
        <div class="data-row"><span class="data-label">Pending</span><span id="fwdPending" class="data-value">0</span></div>
      </div>
      <div id="fwdTrades" style="max-height:180px;overflow-y:auto;"></div>
    </div>

    <div class="countdown">Next refresh in <span id="countdown">{{ refresh }}</span>s</div>
  </div>
</div>

<script>
const DIGITS = {{ digits }};
const REFRESH = {{ refresh }};
let countdownVal = REFRESH;

const darkLayout = {
  paper_bgcolor: '#0d1117', plot_bgcolor: '#16213e',
  font: { color: '#e0e0e0', family: 'JetBrains Mono, monospace', size: 10 },
  margin: { l: 60, r: 20, t: 10, b: 0 },
  xaxis: { gridcolor: '#2a2a4a', showgrid: true, rangeslider: { visible: false }, type: 'category', dtick: 15, tickangle: -30, tickfont: { size: 8 } },
  yaxis: { gridcolor: '#2a2a4a', showgrid: true, side: 'right', tickformat: '.' + DIGITS + 'f' },
  showlegend: false, dragmode: 'pan',
};
const volLayout = {
  paper_bgcolor: '#0d1117', plot_bgcolor: '#16213e',
  font: { color: '#888', family: 'JetBrains Mono', size: 8 },
  margin: { l: 60, r: 20, t: 0, b: 30 },
  xaxis: { gridcolor: '#2a2a4a', type: 'category', dtick: 15, tickangle: -30, tickfont: { size: 8 } },
  yaxis: { gridcolor: '#2a2a4a', showgrid: true, side: 'right' },
  showlegend: false, bargap: 0.2,
};

function formatTime(t) {
  let d = new Date(t.replace(' ', 'T'));
  if (isNaN(d.getTime())) { d = new Date(t); } // Fallback
  return (d.getMonth()+1).toString().padStart(2,'0') + '/' + d.getDate().toString().padStart(2,'0') + ' ' + d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
}

let lastGoodData = null;

function updateDashboard(data) {
  // Show warning/error banner
  let statusDot = document.querySelector('.status-dot');
  if (data.error && !data.candles) {
    // Total failure — use cached data if available
    console.error('API error:', data.error);
    if (lastGoodData && lastGoodData.candles) {
      data = lastGoodData;
      data.warning = 'Using cached data — ' + (data.warning || 'live data unavailable');
    } else {
      document.getElementById('updateTime').textContent = 'ERROR: ' + data.error;
      statusDot.style.background = '#ff3366';
      return;
    }
  }

  if (data.warning) {
    document.getElementById('updateTime').textContent = data.timestamp + '  ⚠ ' + data.warning;
    statusDot.style.background = '#ffaa00';
  } else {
    document.getElementById('updateTime').textContent = data.timestamp;
    statusDot.style.background = '#00ff88';
  }

  // --- Charts ---
  let c = data.candles;
  if (c && c.time && c.time.length > 0) {
    let labels = c.time.map(formatTime);
    let candleTrace = { x: labels, open: c.open, high: c.high, low: c.low, close: c.close, type: 'candlestick', increasing: { line: { color: '#00d4aa' }, fillcolor: '#00d4aa' }, decreasing: { line: { color: '#ff4757' }, fillcolor: '#ff4757' } };

    let shapes = []; let annotations = [];
    (data.signals || []).forEach(s => {
      let idx = labels.indexOf(formatTime(s.time));
      if (idx < 0) return;
      annotations.push({ x: labels[idx], y: s.signal === 'BUY' ? c.low[idx] : c.high[idx], text: s.signal === 'BUY' ? '▲' : '▼', font: { color: s.signal === 'BUY' ? '#00ff88' : '#ff3366', size: 16 }, showarrow: false, yshift: s.signal === 'BUY' ? -12 : 12 });
      let endIdx = Math.min(idx + 15, labels.length - 1);
      shapes.push({ type: 'line', x0: labels[idx], x1: labels[endIdx], y0: s.tp, y1: s.tp, line: { color: '#00ff88', width: 1, dash: 'dash' }, opacity: 0.6 });
      shapes.push({ type: 'line', x0: labels[idx], x1: labels[endIdx], y0: s.sl, y1: s.sl, line: { color: '#ff3366', width: 1, dash: 'dash' }, opacity: 0.6 });
      annotations.push({ x: labels[endIdx], y: s.tp, text: 'TP ' + s.tp.toFixed(DIGITS), font: { color: '#00ff88', size: 8 }, showarrow: false, xanchor: 'left', xshift: 4 });
      annotations.push({ x: labels[endIdx], y: s.sl, text: 'SL ' + s.sl.toFixed(DIGITS), font: { color: '#ff3366', size: 8 }, showarrow: false, xanchor: 'left', xshift: 4 });
    });

    let lastPrice = c.close[c.close.length - 1];
    shapes.push({ type: 'line', x0: labels[0], x1: labels[labels.length-1], y0: lastPrice, y1: lastPrice, line: { color: '#ffd700', width: 1 }, opacity: 0.5 });
    annotations.push({ x: labels[labels.length-1], y: lastPrice, text: ' ' + lastPrice.toFixed(DIGITS), font: { color: '#ffd700', size: 10 }, showarrow: false, xanchor: 'left', bgcolor: 'rgba(255,215,0,0.15)' });

    Plotly.react('priceChart', [candleTrace], Object.assign({}, darkLayout, { shapes: shapes, annotations: annotations }), { responsive: true, displayModeBar: false });
    Plotly.react('volumeChart', [{ x: labels, y: c.volume, type: 'bar', marker: { color: c.close.map((cl, i) => cl >= c.open[i] ? 'rgba(0,212,170,0.4)' : 'rgba(255,71,87,0.4)') } }], volLayout, { responsive: true, displayModeBar: false });
  }

  // --- Current Signal ---
  let sig = data.current_signal || {signal: 'NONE', confidence: 0};
  let sigBox = document.getElementById('signalBox');
  let sigLabel = document.getElementById('signalLabel');
  let sigConf = document.getElementById('signalConf');
  let sigLevels = document.getElementById('signalLevels');
  let sigClass = sig.signal === 'BUY' ? 'buy' : sig.signal === 'SELL' ? 'sell' : sig.signal === 'HOLD' ? 'hold' : 'none';
  sigBox.className = 'signal-box ' + sigClass; sigLabel.className = 'signal-label ' + sigClass;
  if (sig.signal === 'BUY' || sig.signal === 'SELL') {
    sigLabel.textContent = (sig.signal === 'BUY' ? '▲ ' : '▼ ') + sig.signal; sigConf.textContent = 'Confidence: ' + (sig.confidence * 100).toFixed(1) + '%'; sigLevels.style.display = 'block';
    document.getElementById('sigEntry').textContent = sig.entry.toFixed(DIGITS); document.getElementById('sigSL').textContent = sig.sl.toFixed(DIGITS); document.getElementById('sigTP').textContent = sig.tp.toFixed(DIGITS);
    document.getElementById('sigSLPts').textContent = (sig.sl_pts || 0) + ' pts'; document.getElementById('sigTPPts').textContent = (sig.tp_pts || 0) + ' pts';
  } else {
    let label = sig.signal === 'HOLD' ? '■ HOLD' : sig.signal === 'LOW_CONF' ? '○ LOW CONF' : sig.signal === 'MARKET_CLOSED' ? '◆ MARKET CLOSED' : '— WAIT';
    sigLabel.textContent = label; sigConf.textContent = sig.confidence > 0 ? 'Conf: ' + (sig.confidence * 100).toFixed(1) + '%' : 'No signal'; sigLevels.style.display = 'none';
  }

  // --- Tick (with null checks) ---
  if (data.tick && data.tick.bid != null) {
    document.getElementById('bid').textContent = data.tick.bid.toFixed(DIGITS);
    document.getElementById('ask').textContent = data.tick.ask.toFixed(DIGITS);
    document.getElementById('spread').textContent = data.tick.spread + ' pts';
  } else {
    document.getElementById('bid').textContent = '—';
    document.getElementById('ask').textContent = '—';
    document.getElementById('spread').textContent = '—';
  }

  // --- Account ---
  if (data.account && data.account.balance) {
    document.getElementById('balance').textContent = '$' + data.account.balance.toLocaleString();
    document.getElementById('equity').textContent = '$' + data.account.equity.toLocaleString();
    let fl = data.account.floating; document.getElementById('floating').textContent = (fl >= 0 ? '+' : '') + '$' + fl.toFixed(2); document.getElementById('floating').className = 'data-value ' + (fl >= 0 ? 'green' : 'red');
  }

  // --- Positions ---
  let positions = data.positions || [];
  document.getElementById('posCount').textContent = '(' + positions.length + ')';
  document.getElementById('positionsContainer').innerHTML = positions.length === 0 ? '<span style="color:#555;font-size:12px;">No positions</span>' : positions.map(p => `<div class="pos-card ${p.type.toLowerCase()}"><div class="pos-header"><span class="pos-type ${p.type.toLowerCase()}">${p.type} ${p.volume}</span><span class="data-value ${p.profit >= 0 ? 'green' : 'red'}">${p.profit >= 0 ? '+' : ''}$${p.profit.toFixed(2)}</span></div><div style="color:#888;font-size:10px;">#${p.ticket} @ ${p.open_price.toFixed(DIGITS)} | SL ${p.sl.toFixed(DIGITS)} TP ${p.tp.toFixed(DIGITS)}</div></div>`).join('');

  // --- MT5 Live History ---
  if (data.mt5_history) {
    let mt5 = data.mt5_history;
    document.getElementById('mt5Total').textContent = mt5.total;
    document.getElementById('mt5WL').textContent = mt5.wins + ' / ' + mt5.losses;
    document.getElementById('mt5WR').textContent = mt5.win_rate.toFixed(1) + '%';
    document.getElementById('mt5Profit').textContent = (mt5.profit >= 0 ? '+' : '') + '$' + mt5.profit.toFixed(2);
    document.getElementById('mt5Profit').className = 'data-value ' + (mt5.profit >= 0 ? 'green' : 'red');
    
    document.getElementById('mt5Trades').innerHTML = mt5.history.length === 0 ? '<span style="color:#555;font-size:11px;">No live trades yet.</span>' : mt5.history.reverse().map(t => `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1a1a2e;font-size:10px;"><span style="color:#888;">${t.time}</span><span style="color:${t.type==='BUY'?'#00ff88':'#ff3366'};">${t.type}</span><span style="color:${t.profit>=0?'#00ff88':'#ff3366'};">${t.profit>=0?'+':''}$${t.profit.toFixed(2)}</span></div>`).join('');
  }

  // --- Live Trade Status ---
  let btn = document.getElementById('toggleTradeBtn');
  if (data.live_trade_enabled) {
    btn.textContent = 'Live Trade: ON';
    btn.classList.add('active');
  } else {
    btn.textContent = 'Live Trade: OFF';
    btn.classList.remove('active');
  }

  // --- Forward Test ---
  if (data.forward_test) {
    let ft = data.forward_test;
    document.getElementById('fwdTotal').textContent = ft.total_trades; document.getElementById('fwdWL').textContent = ft.wins + ' / ' + ft.losses + ' / ' + ft.expired;
    document.getElementById('fwdWR').textContent = ft.win_rate.toFixed(1) + '%'; document.getElementById('fwdPts').textContent = (ft.total_points >= 0 ? '+' : '') + ft.total_points + ' pts';
    document.getElementById('fwdPending').textContent = ft.pending;
    let allD = [...(ft.open_trades || []).map(t => ({...t, result: 'OPEN', points: '—'})), ...(ft.trades || []).slice().reverse()];
    document.getElementById('fwdTrades').innerHTML = allD.length === 0 ? '<span style="color:#555;font-size:11px;">No trades.</span>' : allD.slice(0, 15).map(t => `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1a1a2e;font-size:10px;"><span style="color:#888;">${formatTime(t.time || t.entry_time)}</span><span style="color:${t.signal==='BUY'?'#00ff88':'#ff3366'};">${t.signal}</span><span style="color:${t.result==='WIN'?'#00ff88':t.result==='LOSS'?'#ff3366':'#ffd700'};">${t.result}</span><span>${t.points}</span></div>`).join('');
  }

  // Cache successful data
  if (data.candles) lastGoodData = data;
}

async function fetchData() { try { let r = await fetch('/api/data'); let d = await r.json(); updateDashboard(d); } catch (e) { console.error(e); updateDashboard({error: 'Network error: ' + e.message}); } }
async function toggleLiveTrade() { 
  try { 
    let r = await fetch('/api/toggle_trade', {method: 'POST'}); 
    let d = await r.json(); 
    fetchData(); // refresh UI immediately
  } catch (e) { console.error('Toggle error', e); } 
}
function startCountdown() { countdownVal = REFRESH; document.getElementById('countdown').textContent = countdownVal; let t = setInterval(() => { countdownVal--; document.getElementById('countdown').textContent = Math.max(0, countdownVal); if (countdownVal <= 0) { clearInterval(t); fetchData().then(() => startCountdown()); } }, 1000); }
fetchData().then(() => startCountdown());
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, symbol=SYMBOL, timeframe=TIMEFRAME_NAME, digits=DIGITS, refresh=REFRESH_SECONDS)

@app.route('/api/data')
def api_data():
    return jsonify(get_dashboard_data())

@app.route('/api/toggle_trade', methods=['POST'])
def toggle_trade():
    global live_trade_enabled
    with live_trade_lock:
        live_trade_enabled = not live_trade_enabled
    return jsonify({'status': 'success', 'live_trade_enabled': live_trade_enabled})

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if not mt5.initialize():
        print("MT5 FAILED"); sys.exit(1)
    
    sym_info = mt5.symbol_info(SYMBOL)
    if not sym_info: print(f"{SYMBOL} NOT FOUND"); sys.exit(1)
    DIGITS, POINT = sym_info.digits, sym_info.point

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    print(f"Starting dashboard on http://127.0.0.1:{PORT} ...")
    # threaded=False is REQUIRED! MetaTrader5 python module throws silent errors 
    # (returns None for all operations) if accessed from uninitialized side-threads
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False, threaded=False)