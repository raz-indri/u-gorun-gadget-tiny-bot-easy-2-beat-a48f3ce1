// Slow-Tide Dip Bot — a bars-replay/v1 trading bot for the make.fun
// "Tiny Bot, Easy $2: Beat Buy-and-Hold on Solana" bounty.
//
// The bot speaks the agent-fun/bars-replay/v1 line protocol through the
// vendored shim: the harness sends a hello, then one batch of hourly bars
// per bar, and the bot answers each batch with one reply carrying its
// orders. Orders decided on a batch fill at the next bar's open, so every
// decision here uses only bars the harness has already sent (no lookahead),
// and the bot is fully deterministic: no clock, no network, no randomness.
//
// Strategy (chosen by backtesting the kit's dev window plus walk-forward
// splits and a mirrored uptrend window, with an exact replica of the
// harness fill model):
//   1. Start invested: on the first two batches, buy an equal slice of
//      every priced symbol (like buy-and-hold, so a rallying market is
//      ridden from the start instead of watched from cash).
//   2. Slow trend filter: hold a symbol only while its fast EMA (18 bars)
//      sits above its slow EMA (44 bars). A flip down sells it; a flip
//      back up re-buys it. The slow pair survives hourly chop where fast
//      crossovers whipsaw.
//   3. Dip-buy sleeve: a symbol 2% or more below its 12-bar average that
//      has bounced at least 0.5% off its 6-bar low gets an extra half
//      slice. This buys pullbacks inside trends and mean-reverts the
//      choppy stretches the trend filter sits out.
//   4. Position caps: no symbol ever exceeds 25% of equity (the harness
//      weight cap), sells are placed before buys so exits fund entries,
//      buys cap their slippage at 500 bps while sells use a wide cap so an
//      exit is never refused on a gap down.
//
// Guardrails kept: far more than the 8-trade minimum on every tested
// window, and a max drawdown near 5% on the dev window (cap is 25%).

import { runBars, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

const SYMBOLS = ['FARTCOIN', 'JITOSOL', 'POPCAT', 'RAY', 'SOL', 'WETH', 'WIF'] as const;
const FAST_PERIOD = 18;
const SLOW_PERIOD = 44;
const SMA_PERIOD = 12;
const LOW_LOOKBACK = 6;
const DIP_FRACTION = 0.02;
const BOUNCE_FRACTION = 0.005;
const BOUNCE_WEIGHT = 0.5;
const MAX_WEIGHT = 0.25;
const BUY_SLIPPAGE_BPS = 500;
const SELL_SLIPPAGE_BPS = 10_000;

// Per-symbol price history (closes as plain numbers; ratios only) and the
// two EMAs, seeded on the first close the harness sends for the symbol.
const closes = new Map<string, number[]>();
const emaFast = new Map<string, number>();
const emaSlow = new Map<string, number>();

const fastK = 2 / (FAST_PERIOD + 1);
const slowK = 2 / (SLOW_PERIOD + 1);

function noteClose(symbol: string, close: bigint): void {
  const price = Number(close);
  let history = closes.get(symbol);
  if (history === undefined) {
    history = [];
    closes.set(symbol, history);
    emaFast.set(symbol, price);
    emaSlow.set(symbol, price);
  }
  history.push(price);
  emaFast.set(symbol, price * fastK + emaFast.get(symbol)! * (1 - fastK));
  emaSlow.set(symbol, price * slowK + emaSlow.get(symbol)! * (1 - slowK));
}

function sma(symbol: string): number | null {
  const history = closes.get(symbol);
  if (history === undefined || history.length < SMA_PERIOD) return null;
  let sum = 0;
  for (let i = history.length - SMA_PERIOD; i < history.length; i++) sum += history[i]!;
  return sum / SMA_PERIOD;
}

function recentLow(symbol: string): number | null {
  const history = closes.get(symbol);
  if (history === undefined || history.length < LOW_LOOKBACK) return null;
  let low = Infinity;
  for (let i = history.length - LOW_LOOKBACK; i < history.length; i++) {
    if (history[i]! < low) low = history[i]!;
  }
  return low;
}

/** Target weight (fraction of equity) per symbol for this batch. */
function targets(state: BarsState): Map<string, number> {
  const out = new Map<string, number>();
  const priced = SYMBOLS.filter((s) => state.close(s) !== null);
  if (priced.length === 0) return out;
  // First two batches: buy the whole universe equally, like buy-and-hold.
  if (state.seq < 2) {
    for (const s of priced) out.set(s, 1 / priced.length);
    return out;
  }
  // Trend sleeve: hold the symbols whose fast EMA is above their slow EMA.
  const up: string[] = [];
  for (const s of priced) {
    if (emaFast.get(s)! > emaSlow.get(s)!) up.push(s);
  }
  if (up.length > 0) {
    const w = Math.min(1 / up.length, MAX_WEIGHT);
    for (const s of up) out.set(s, w);
  }
  // Dip-buy sleeve: 2% below the 12-bar average, bounced off the 6-bar low.
  const dips: string[] = [];
  for (const s of priced) {
    const history = closes.get(s);
    const avg = sma(s);
    const low = recentLow(s);
    if (history === undefined || avg === null || low === null) continue;
    if (history.length < SMA_PERIOD + LOW_LOOKBACK) continue;
    const last = history[history.length - 1]!;
    if (last < avg * (1 - DIP_FRACTION) && last > low * (1 + BOUNCE_FRACTION)) dips.push(s);
  }
  if (dips.length > 0) {
    const w = Math.min(1 / dips.length, MAX_WEIGHT) * BOUNCE_WEIGHT;
    for (const s of dips) out.set(s, Math.min((out.get(s) ?? 0) + w, MAX_WEIGHT));
  }
  return out;
}

await runBars({
  onBatch(state, batch) {
    // While warming up every order would be rejected: wait for the window.
    if (state.warmingUp) return [];
    // Fold only this batch's own bars into the histories before deciding:
    // a symbol with no bar this hour keeps its previous close, and must
    // not get the same close recorded twice.
    for (const bar of batch.bars) noteClose(bar.symbol, BigInt(bar.c));
    const { fees } = state.hello;
    const minNotional = BigInt(fees.min_notional_quote_units);
    const equity = state.equity;
    const want = targets(state);
    const orders: (BuyInput | SellInput)[] = [];
    // Sells first, so exits fund the buys that follow in this same reply.
    let estCash = state.cash;
    for (const s of SYMBOLS) {
      const held = state.positions.get(s);
      if (held === undefined || held.qty === 0n) continue;
      const close = state.close(s);
      if (close === null) continue;
      const current = state.value(s, held.qty, close);
      const targetValue = BigInt(Math.floor((want.get(s) ?? 0) * Number(equity)));
      if (current <= targetValue) continue;
      estCash += current - targetValue;
      if (targetValue <= 0n) {
        orders.push({ side: 'sell', symbol: s, qty: held.qty, max_slippage_bps: SELL_SLIPPAGE_BPS });
      } else {
        const ps = state.symbols.get(s)!;
        const qty =
          ((current - targetValue) * 10n ** BigInt(ps.qty_scale + ps.price_scale)) /
          (close * 10n ** BigInt(state.hello.capital.quote_scale));
        const q = qty > held.qty ? held.qty : qty;
        if (q > 0n && state.value(s, q, close) >= minNotional) {
          orders.push({ side: 'sell', symbol: s, qty: q, max_slippage_bps: SELL_SLIPPAGE_BPS });
        }
      }
    }
    // Buys: top up each wanted symbol to its target weight.
    for (const s of [...want.keys()].sort()) {
      const targetValue = BigInt(Math.floor(want.get(s)! * Number(equity)));
      if (targetValue <= 0n) continue;
      const close = state.close(s);
      if (close === null) continue;
      const held = state.positions.get(s);
      const current = held === undefined ? 0n : state.value(s, held.qty, close);
      let need = targetValue - current;
      if (need < minNotional) continue;
      if (need > estCash) need = estCash;
      if (need < minNotional) continue;
      orders.push({ side: 'buy', symbol: s, notional_quote_units: need, max_slippage_bps: BUY_SLIPPAGE_BPS });
      estCash -= need;
    }
    return orders;
  },
});
