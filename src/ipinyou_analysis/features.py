from __future__ import annotations

import re

import numpy as np
import pandas as pd


def clean_ip_prefix(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    return re.sub(r"\*$", "", str(value))


def parse_user_agent(user_agent: str) -> tuple[str, str]:
    if not isinstance(user_agent, str) or not user_agent.strip():
        return "unknown", "unknown"

    value = user_agent.lower()

    if "android" in value:
        device_type = "android"
    elif "iphone" in value or "ipad" in value or "ios" in value:
        device_type = "ios"
    elif "windows" in value:
        device_type = "windows"
    elif "mac os" in value or "macintosh" in value:
        device_type = "mac"
    elif "linux" in value:
        device_type = "linux"
    else:
        device_type = "other"

    if "chrome" in value:
        browser = "chrome"
    elif "firefox" in value:
        browser = "firefox"
    elif "msie" in value or "trident" in value:
        browser = "internet_explorer"
    elif "safari" in value:
        browser = "safari"
    elif "opera" in value:
        browser = "opera"
    else:
        browser = "other"

    return device_type, browser


def count_user_tags(user_tag_value: str) -> int:
    if not isinstance(user_tag_value, str) or not user_tag_value.strip():
        return 0
    return len([tag for tag in user_tag_value.split(",") if tag])


def infer_url_presence(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return int(str(value).strip().lower() not in {"", "nan", "null"})


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    if "click" not in frame.columns:
        if "logtype" in frame.columns:
            frame["click"] = (pd.to_numeric(frame["logtype"], errors="coerce") == 2).astype(int)
        else:
            raise ValueError("The dataset does not include a click signal.")

    if "payprice" in frame.columns:
        frame["won_auction"] = (pd.to_numeric(frame["payprice"], errors="coerce").fillna(0) > 0).astype(int)
    elif "impression" in frame.columns:
        frame["won_auction"] = pd.to_numeric(frame["impression"], errors="coerce").fillna(0).astype(int)

    if "slotwidth" in frame.columns and "slotheight" in frame.columns:
        width = pd.to_numeric(frame["slotwidth"], errors="coerce").fillna(0)
        height = pd.to_numeric(frame["slotheight"], errors="coerce").fillna(0)
        frame["slot_area"] = width * height
        frame["slot_aspect_ratio"] = np.where(height > 0, width / height, np.nan)

    if "bidprice" in frame.columns and "payprice" in frame.columns:
        bidprice = pd.to_numeric(frame["bidprice"], errors="coerce")
        payprice = pd.to_numeric(frame["payprice"], errors="coerce")
        frame["bid_gap"] = bidprice - payprice
        frame["bid_to_pay_ratio"] = np.where(payprice > 0, bidprice / payprice, np.nan)

    if "slotprice" in frame.columns and "payprice" in frame.columns:
        slotprice = pd.to_numeric(frame["slotprice"], errors="coerce")
        payprice = pd.to_numeric(frame["payprice"], errors="coerce")
        frame["floor_gap"] = payprice - slotprice

    if "useragent" in frame.columns:
        parsed = frame["useragent"].apply(parse_user_agent)
        frame["device_type"] = parsed.map(lambda item: item[0])
        frame["browser"] = parsed.map(lambda item: item[1])

    if "usertag" in frame.columns:
        frame["user_tag_count"] = frame["usertag"].fillna("").map(count_user_tags)

    frame["has_domain"] = frame.get("domain", "").map(infer_url_presence) if "domain" in frame.columns else 0
    frame["has_url"] = frame.get("url", "").map(infer_url_presence) if "url" in frame.columns else 0
    frame["has_urlid"] = frame.get("urlid", "").map(infer_url_presence) if "urlid" in frame.columns else 0

    if "weekday_name" not in frame.columns and "weekday" in frame.columns:
        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        frame["weekday_name"] = frame["weekday"].map(weekday_map)

    if "keypage" in frame.columns:
        frame["keypage_prefix"] = frame["keypage"].fillna("").astype(str).str.extract(r"(^.{0,12})", expand=False)

    if "ip" in frame.columns and "ip_prefix" not in frame.columns:
        frame["ip_prefix"] = frame["ip"].map(clean_ip_prefix)

    if "weekday" in frame.columns:
        frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype(int)

    return frame
