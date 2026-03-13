"""
XGBoost Trading - Lightweight Web Dashboard
============================================
Reads pre-computed data from latest_state.json (written by engine.py).
No MT5 dependency — responds instantly.

Run engine.py separately for data computation:
  python engine.py

Then start this dashboard:
  python dashboard1.py
"""

import os, json
from datetime import datetime
from flask import Flask, jsonify, render_template_string

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "XAUUSDm"
TIMEFRAME_NAME = "M15"
DIGITS = 3
POINT = 0.01
REFRESH_SECONDS = 15   # Must match engine.py REFRESH_SECONDS
PORT = 5555

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_BASE_DIR, "latest_state.json")
LIVE_TRADE_CONFIG = os.path.join(_BASE_DIR, "live_trade_config.json")

# ============================================================
# CACHE READING
# ============================================================
def read_cache():
    """Read latest engine state from JSON cache file."""
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return None

def _read_live_trade_config():
    """Read live trade enabled state from shared config file."""
    try:
        if os.path.exists(LIVE_TRADE_CONFIG):
            with open(LIVE_TRADE_CONFIG, 'r') as f:
                return json.load(f).get('enabled', False)
    except Exception:
        pass
    return False

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
    document.getElementById('fwdTrades').innerHTML = allD.length === 0 ? '<span style="color:#555;font-size:11px;">No trades.</span>' : allD.slice(0, 15).map(t => `<div style="display:flex;flex-wrap:wrap;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1a1a2e;font-size:10px;"><span style="color:#888;">${formatTime(t.time || t.entry_time)}</span><span style="color:${t.signal==='BUY'?'#00ff88':'#ff3366'};">${t.signal}</span><span style="color:${t.result==='WIN'?'#00ff88':t.result==='LOSS'?'#ff3366':'#ffd700'};">${t.result}</span><span>${t.points}</span>${t.sl != null && t.tp != null && t.entry != null ? `<div style="width:100%;display:flex;justify-content:space-between;font-size:9px;color:#888;"><span>SL <span style="color:#ff3366;">${t.sl.toFixed(DIGITS)}</span></span><span>TP <span style="color:#00ff88;">${t.tp.toFixed(DIGITS)}</span></span><span>Entry <span style="color:#ffd700;">${t.entry.toFixed(DIGITS)}</span></span></div>` : ''}</div>`).join('');
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
    cache = read_cache()
    digits = (cache or {}).get('digits', DIGITS)
    return render_template_string(HTML_TEMPLATE, symbol=SYMBOL, timeframe=TIMEFRAME_NAME, digits=digits, refresh=REFRESH_SECONDS)

@app.route('/api/data')
def api_data():
    """Return cached data from engine.py — responds instantly."""
    data = read_cache()
    if data is None:
        return jsonify({
            'symbol': SYMBOL, 'timeframe': TIMEFRAME_NAME,
            'digits': DIGITS, 'point': POINT,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'candles': None, 'signals': [],
            'current_signal': {'signal': 'NONE', 'confidence': 0, 'entry': 0, 'sl': 0, 'tp': 0},
            'tick': {}, 'account': {}, 'positions': [],
            'forward_test': {'total_trades': 0, 'wins': 0, 'losses': 0, 'expired': 0,
                           'win_rate': 0.0, 'total_points': 0, 'pending': 0,
                           'trades': [], 'open_trades': []},
            'mt5_history': {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                          'profit': 0.0, 'history': []},
            'live_trade_enabled': False,
            'warning': 'Engine not running. Start engine.py first.',
        })
    # Patch live_trade_enabled from config for immediate responsiveness
    data['live_trade_enabled'] = _read_live_trade_config()
    return jsonify(data)

@app.route('/api/toggle_trade', methods=["POST"])
def toggle_trade():
    """Toggle live trading by writing to shared config file."""
    enabled = _read_live_trade_config()
    enabled = not enabled
    try:
        with open(LIVE_TRADE_CONFIG, 'w') as f:
            json.dump({'enabled': enabled}, f)
    except Exception:
        pass
    return jsonify({'status': 'success', 'live_trade_enabled': enabled})

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("XGB Trading Dashboard (Lightweight)")
    print("=" * 50)
    print(f"Reading data from: {CACHE_PATH}")
    print(f"Make sure engine.py is running separately.")
    print(f"Starting on http://127.0.0.1:{PORT}")
    print("-" * 50)
    # threaded=True is safe now — no MT5 dependency in this process
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False, threaded=True)
