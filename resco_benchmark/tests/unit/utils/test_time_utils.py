import pytest

from utils.time_utils import convert_to_24_hour_format, convert_time_range_to_seconds


class TestConvertTo24HourFormat:

    @pytest.mark.parametrize(
        "hour, meridiem, expected",
        [
            *[(h, "am", 0 if h == 12 else h) for h in range(1, 13)],
            *[(h, "pm", 12 if h == 12 else h + 12) for h in range(1, 13)],
        ],
    )
    def test_valid_conversion(self, hour: int, meridiem: str, expected: int) -> None:
        assert convert_to_24_hour_format(hour, meridiem) == expected

    @pytest.mark.parametrize("hour", [0, 13, -1, 100])
    def test_invalid_hour_range(self, hour: int) -> None:
        with pytest.raises(ValueError, match="Hour must be between 1 and 12."):
            convert_to_24_hour_format(hour, "am")

    @pytest.mark.parametrize("meridiem", ["a.m.", "AM", "morning", "", None])
    def test_invalid_meridiem(self, meridiem: str) -> None:
        with pytest.raises(ValueError, match="Invalid meridiem format"):
            convert_to_24_hour_format(5, meridiem)

    def test_invalid_types(self) -> None:
        with pytest.raises(TypeError):
            convert_to_24_hour_format("5", "am")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            convert_to_24_hour_format(5, 123)  # type: ignore[arg-type]


class TestConvertTimeRangeToSeconds:

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            *[
                (
                    f"{h}-{h + 1}{meridiem}",
                    (
                        convert_to_24_hour_format(h, meridiem) * 3600,
                        convert_to_24_hour_format(h + 1, meridiem) * 3600,
                    ),
                )
                for meridiem in ["am", "pm"]
                for h in range(1, 12)
                if convert_to_24_hour_format(h + 1, meridiem)
                - convert_to_24_hour_format(h, meridiem)
                == 1
            ],
            ("12-1am", (0, 3600)),
            ("12-1pm", (43200, 46800)),
        ],
    )
    def test_valid_time_ranges(self, input_str: str, expected: tuple[int, int]) -> None:
        assert convert_time_range_to_seconds(input_str) == expected

    @pytest.mark.parametrize(
        "input_str", ["7-9am", "12-2pm", "11-1pm", "1-1pm", "12-12am", "10-9pm"]
    )
    def test_invalid_range_length(self, input_str: str) -> None:
        with pytest.raises(ValueError) as exc:
            convert_time_range_to_seconds(input_str)
        assert "Invalid time range" in str(exc.value)

    @pytest.mark.parametrize(
        "input_str",
        [
            "7am-8am",
            "7-8",
            "07-08am",
            "01-02pm",
            "7 - 8am",
            "3-4Pm",
            "7_8am",
            "1_2AM",
            "3_4PM",
            " 3_4am",
            "seven-eightam",
        ],
    )
    def test_invalid_format(self, input_str: str) -> None:
        with pytest.raises(ValueError) as exc:
            convert_time_range_to_seconds(input_str)
        assert "Invalid time format" in str(exc.value)
