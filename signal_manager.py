"""
Aboud Trading Bot - Signal Manager v4.0 (Pocket Option Native)
==============================================================
Analyzes Pocket Option data directly and manages trade lifecycle.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta

from config import (
    SIGNAL_CONFIRM_MIN_SECONDS,
    SIGNAL_CONFIRM_MAX_SECONDS,
    SIGNAL_CONFIRM_CHECK_INTERVAL,
    TRADE_DURATION_MINUTES,
    TRADING_PAIRS,
    TRADING_START_HOUR_UTC,
    TRADING_END_HOUR_UTC,
    BOT_TIMEZONE,
    RESULT_CANDLE_BUFFER_SECONDS,
    RESULT_FETCH_RETRY_SECONDS,
    RESULT_MAX_WAIT_AFTER_EXPIRY_SECONDS,
    PO_ANALYSIS_INTERVAL,
)
from database import (
    create_pending_signal,
    confirm_pending_signal,
    cancel_pending_signal,
    create_trade,
    update_trade_entry_price,
    update_trade_result,
    update_statistics,
    is_signals_enabled,
    get_pair_statistics,
)
from pocket_option_service import get_analyzer, get_data_service, get_asset_names

logger = logging.getLogger(__name__)


class SignalManager:
    def __init__(self, telegram_sender):
        self.telegram = telegram_sender
        self.active_pending = {}
        self.pending_results = {}
        self.active_trade = None
        self.active_trade_lock = asyncio.Lock()
        self._last_signal = {}
        self._analysis_task = None

    def start_analysis(self):
        """Start the background analysis loop."""
        if self._analysis_task is None or self._analysis_task.done():
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            logger.info("Background analysis loop started.")

    async def _analysis_loop(self):
        """Main loop that checks for signals every interval."""
        while True:
            try:
                if is_signals_enabled() and self.is_trading_hours():
                    for pair in TRADING_PAIRS:
                        if self.has_active_trade() and self.active_trade["pair"] == pair:
                            continue
                        
                        if pair in self.active_pending:
                            continue

                        await self._check_pair_for_signal(pair)
                
                await asyncio.sleep(PO_ANALYSIS_INTERVAL)
            except Exception as e:
                logger.error("Error in analysis loop: %s", e, exc_info=True)
                await asyncio.sleep(10)

    async def _check_pair_for_signal(self, pair):
        """Fetch data and analyze a specific pair."""
        data_service = get_data_service()
        analyzer = get_analyzer()
        
        if not data_service:
            return

        assets = get_asset_names(pair)
        for asset in assets:
            candles = await data_service.get_candles(asset, timeframe_seconds=900, count=100)
            if candles:
                signal = analyzer.analyze(candles)
                if signal:
                    logger.info("Signal detected for %s (%s): %s", pair, asset, signal["direction"])
                    # Process as if it came from a webhook
                    data = {
                        "pair": pair,
                        "direction": signal["direction"],
                        "action": "SIGNAL",
                        "indicators": signal["indicators"],
                        "target_entry_time": self.get_next_candle_time().isoformat()
                    }
                    await self.process_webhook_signal(data)
                    break # Only one signal per pair

    def is_trading_hours(self):
        if TRADING_START_HOUR_UTC == 0 and TRADING_END_HOUR_UTC == 24:
            return True
        now = datetime.now(timezone.utc)
        return TRADING_START_HOUR_UTC <= now.hour < TRADING_END_HOUR_UTC

    def is_valid_pair(self, pair):
        return pair.upper().replace("/", "") in TRADING_PAIRS

    def get_next_candle_time(self, now=None):
        now = now or datetime.now(timezone.utc)
        minute = now.minute
        next_slot = ((minute // 15) + 1) * 15
        if next_slot >= 60:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return now.replace(minute=next_slot, second=0, microsecond=0)

    def utc_to_local(self, dt):
        return dt.astimezone(BOT_TIMEZONE)

    def has_active_trade(self):
        return self.active_trade is not None

    async def process_webhook_signal(self, data):
        """Main entry point for signals (both internal and external)."""
        pair = data.get("pair", "").upper().replace("/", "")
        direction = data.get("direction", "").upper()
        action = data.get("action", "SIGNAL").upper()
        indicators = data.get("indicators", {})

        if not is_signals_enabled():
            return {"status": "disabled"}

        if not self.is_valid_pair(pair):
            return {"status": "error", "message": f"Invalid pair: {pair}"}

        if direction not in ["CALL", "PUT"]:
            return {"status": "error", "message": f"Invalid direction: {direction}"}

        if action == "SIGNAL":
            self._last_signal[pair] = {
                "direction": direction,
                "time": datetime.now(timezone.utc),
            }

        if action == "CANCEL":
            self._last_signal.pop(pair, None)
            return await self._cancel_active_pending(pair)

        if self.has_active_trade():
            return {"status": "blocked", "message": "Active trade in progress"}

        if pair in self.active_pending and not self.active_pending[pair].done():
            return {"status": "duplicate", "message": f"Pending signal exists for {pair}"}

        return await self._create_temporary_signal(pair, direction, indicators, data)

    async def _create_temporary_signal(self, pair, direction, indicators, payload):
        now = datetime.now(timezone.utc)
        target_entry = self.get_next_candle_time()
        
        signal_id = create_pending_signal(
            pair=pair,
            direction=direction,
            detected_at=now.isoformat(),
            target_entry_time=target_entry.isoformat(),
            indicator_data=indicators,
        )

        local_entry = self.utc_to_local(target_entry)
        logger.info(
            "Pending #%s: %s %s (entry %s UTC+3)",
            signal_id, pair, direction, local_entry.strftime("%H:%M"),
        )

        if pair in self.active_pending:
            old_task = self.active_pending[pair]
            if not old_task.done():
                old_task.cancel()

        task = asyncio.create_task(
            self._smart_confirmation(signal_id, pair, direction, target_entry, indicators)
        )
        self.active_pending[pair] = task
        return {"status": "pending", "signal_id": signal_id}

    async def _smart_confirmation(self, signal_id, pair, direction, target_entry, indicators):
        try:
            elapsed = 0
            confirmed = False

            while elapsed < SIGNAL_CONFIRM_MAX_SECONDS:
                await asyncio.sleep(SIGNAL_CONFIRM_CHECK_INTERVAL)
                elapsed += SIGNAL_CONFIRM_CHECK_INTERVAL

                if not is_signals_enabled():
                    cancel_pending_signal(signal_id)
                    return

                if self.has_active_trade():
                    cancel_pending_signal(signal_id)
                    self.active_pending.pop(pair, None)
                    return

                # Re-verify signal from Pocket Option
                data_service = get_data_service()
                analyzer = get_analyzer()
                assets = get_asset_names(pair)
                
                signal_still_valid = False
                for asset in assets:
                    candles = await data_service.get_candles(asset, timeframe_seconds=900, count=100)
                    if candles:
                        current_sig = analyzer.analyze(candles)
                        if current_sig and current_sig["direction"] == direction:
                            signal_still_valid = True
                            break
                
                if not signal_still_valid:
                    cancel_pending_signal(signal_id)
                    logger.info("#%s cancelled - signal disappeared for %s", signal_id, pair)
                    self.active_pending.pop(pair, None)
                    return

                if elapsed >= SIGNAL_CONFIRM_MIN_SECONDS:
                    confirmed = True
                    break

            if not confirmed:
                cancel_pending_signal(signal_id)
                self.active_pending.pop(pair, None)
                return

            confirm_pending_signal(signal_id)
            local_entry = self.utc_to_local(target_entry)
            entry_time_str = local_entry.strftime("%H:%M")

            trade_id = create_trade(
                pair=pair,
                direction=direction,
                entry_time=target_entry.isoformat(),
                entry_price=None,
            )

            async with self.active_trade_lock:
                self.active_trade = {
                    "trade_id": trade_id,
                    "pair": pair,
                    "direction": direction,
                    "entry_time": entry_time_str,
                    "target_entry_utc": target_entry,
                }

            stats = get_pair_statistics(pair) or {"total_wins": 0, "total_losses": 0}
            await self.telegram.send_signal(
                pair=pair,
                direction=direction,
                entry_time=entry_time_str,
                stats=stats,
            )

            result_task = asyncio.create_task(
                self._monitor_trade(trade_id, pair, direction, entry_time_str, target_entry)
            )
            self.pending_results[trade_id] = result_task
            self.active_pending.pop(pair, None)

        except asyncio.CancelledError:
            cancel_pending_signal(signal_id)
            self.active_pending.pop(pair, None)
        except Exception as exc:
            logger.error("Error in confirmation #%s: %s", signal_id, exc)
            cancel_pending_signal(signal_id)
            self.active_pending.pop(pair, None)

    async def _monitor_trade(self, trade_id, pair, direction, entry_time_str, target_entry):
        try:
            now = datetime.now(timezone.utc)
            wait_until_entry = (target_entry - now).total_seconds()
            if wait_until_entry > 0:
                await asyncio.sleep(wait_until_entry)

            await asyncio.sleep(RESULT_CANDLE_BUFFER_SECONDS)

            # Get entry price from Pocket Option
            data_service = get_data_service()
            assets = get_asset_names(pair)
            entry_price = None
            
            for asset in assets:
                candles = await data_service.get_candles(asset, timeframe_seconds=900, count=5)
                if candles:
                    entry_price = candles[-1]["open"]
                    break
            
            if entry_price:
                update_trade_entry_price(trade_id, entry_price)

            expiry = target_entry + timedelta(minutes=TRADE_DURATION_MINUTES)
            wait_until_expiry = (expiry - datetime.now(timezone.utc)).total_seconds()
            if wait_until_expiry > 0:
                await asyncio.sleep(wait_until_expiry)

            await asyncio.sleep(RESULT_CANDLE_BUFFER_SECONDS)

            # Determine result from Pocket Option
            exit_price = None
            for asset in assets:
                candles = await data_service.get_candles(asset, timeframe_seconds=900, count=5)
                if candles:
                    exit_price = candles[-1]["close"]
                    break
            
            if entry_price and exit_price:
                if direction == "CALL":
                    result = "WIN" if exit_price > entry_price else "LOSS"
                else:
                    result = "WIN" if exit_price < entry_price else "LOSS"
            else:
                result = "LOSS" # Fallback

            update_trade_result(trade_id, exit_price, result)
            update_statistics(pair, result == "WIN")

            await self.telegram.send_result(
                pair=pair,
                direction=direction,
                entry_time=entry_time_str,
                result=result,
            )

        except Exception as exc:
            logger.error("Error in trade #%s: %s", trade_id, exc)
        finally:
            async with self.active_trade_lock:
                self.active_trade = None
            self.pending_results.pop(trade_id, None)

    async def _cancel_active_pending(self, pair):
        if pair in self.active_pending:
            task = self.active_pending[pair]
            if not task.done():
                task.cancel()
                return {"status": "cancelled"}
        return {"status": "no_pending"}
