import re


def convert_time_range_to_seconds(time_range: str) -> tuple[int, int]:
    """
    Converts a time range string like '7-8am' or '12-1pm' into a tuple of seconds since
     midnight.

    Args:
        time_range (str): Time range in format 'H-Ham' or 'H-Hpm'.

    Returns:
        tuple[int, int]: (start_seconds, end_seconds)

    Raises:
        ValueError: If format is invalid or range is not exactly one hour.
    """
    match = re.fullmatch(r"(1[0-2]|[1-9])-(1[0-2]|[1-9])(am|pm)", time_range)
    if not match:
        raise ValueError(
            "Invalid time format. Use exact format like '7-8am' or '12-1pm' — no "
            "spaces, no uppercase."
        )

    start_hour_str, end_hour_str, meridiem = match.groups()
    start_hour = int(start_hour_str)
    end_hour = int(end_hour_str)

    start_hour_24h = convert_to_24_hour_format(start_hour, meridiem)
    end_hour_24h = convert_to_24_hour_format(end_hour, meridiem)

    if end_hour_24h - start_hour_24h != 1:
        raise ValueError(
            "Invalid time range. Only one-hour ranges like '7-8am' or '12-1pm' are"
            " allowed."
        )

    return start_hour_24h * 3600, end_hour_24h * 3600


def convert_to_24_hour_format(hour: int, meridiem: str) -> int:
    """
    Converts a 12-hour format hour to 24-hour format.

    Args:
        hour (int): Hour in 12-hour format (1–12)
        meridiem (str): 'am' or 'pm'

    Returns:
        int: Hour in 24-hour format

    Raises:
        ValueError: If meridiem is invalid or hour is out of range
    """
    if not 1 <= hour <= 12:
        raise ValueError("Hour must be between 1 and 12.")

    if meridiem == "am":
        return 0 if hour == 12 else hour
    if meridiem == "pm":
        return 12 if hour == 12 else hour + 12

    raise ValueError("Invalid meridiem format. Expected 'am' or 'pm'.")
