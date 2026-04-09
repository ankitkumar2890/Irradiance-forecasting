from __future__ import annotations

import pandas as pd

LOCAL_TIMEZONE = "Asia/Kolkata"


def to_ist_series(values) -> pd.Series:
    dt = pd.Series(pd.to_datetime(values))
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(LOCAL_TIMEZONE)
    return dt.dt.tz_convert(LOCAL_TIMEZONE)
