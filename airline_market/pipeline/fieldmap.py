"""Header normalization adapted from RideResponse's loose-row reader."""

import re
from ..core.errors import DataValidationError


def field_name(x):
    """
    Normalize header punctuation and case without inventing missing fields (model §§1–3,
    calibration assumptions).
    """
    return re.sub("[^a-z0-9]", "", str(x).lower())


class FieldMap:
    """
    Explicit source-header aliases with required-field validation (model §§1–3,
    calibration assumptions).
    """

    def __init__(self, aliases):
        """
        Bind declared inputs and initialize FieldMap bookkeeping (model §§1–3,
        calibration assumptions).
        """
        self.aliases = aliases

    def pick(self, row, logical, required=True):
        """
        Resolve one logical field through the declared aliases (model §§1–3, calibration
        assumptions).
        """
        normalized = {field_name(k): v for k, v in row.items()}
        for name in (logical, *self.aliases.get(logical, ())):
            if field_name(name) in normalized:
                return normalized[field_name(name)]
        if required:
            raise DataValidationError(
                f"missing {logical}; available headers: {list(row)}"
            )
        return None

    def columns(self, headers, required=()):
        """
        Map available headers and reject absent required source fields (model §§1–3,
        calibration assumptions).
        """
        mapping = {}
        for logical in self.aliases:
            matches = [
                h
                for h in headers
                if field_name(h)
                in {field_name(x) for x in (logical, *self.aliases[logical])}
            ]
            if matches:
                mapping[matches[0]] = logical
            elif logical in required:
                raise DataValidationError(
                    f"missing {logical}; available headers: {list(headers)}"
                )
        return mapping


COMMON = {
    "year": [],
    "month": [],
    "quarter": [],
    "carrier": ["UNIQUE_CARRIER", "CARRIER", "OP_UNIQUE_CARRIER", "Reporting_Airline"],
    "origin": ["ORIGIN"],
    "dest": ["DEST"],
    "passengers": ["PASSENGERS"],
    "distance": ["MktDistance", "DISTANCE", "MktMilesFlown"],
}
MAPS = {
    "t100": {
        **COMMON,
        "departures_scheduled": ["DepScheduled"],
        "departures_performed": ["DepPerformed"],
        "seats": [],
        "aircraft_type": [],
    },
    "db1b": {
        **COMMON,
        "itin_id": ["ItinID", "ITIN_ID"],
        "mkt_id": ["MktID"],
        "ticket_carrier": ["TkCarrier", "TICKET_CARRIER"],
        "operating_carrier": ["OpCarrier"],
        "market_fare": ["MktFare", "MARKET_FARE"],
        "market_coupons": ["MktCoupons"],
        "nonstop_miles": ["NonStopMiles"],
    },
    "coupon": {
        **COMMON,
        "itin_id": ["ItinID"],
        "mkt_id": ["MktID"],
        "seq_num": ["SeqNum"],
        "ticket_carrier": ["TkCarrier"],
        "operating_carrier": ["OpCarrier"],
        "fare_class": ["FareClass"],
    },
    "ontime": {
        **COMMON,
        "flight_date": ["FL_DATE", "FlightDate"],
        "flight_number": ["OP_CARRIER_FL_NUM", "Flight_Number_Reporting_Airline"],
        "crs_dep_time": ["CRSDepTime"],
        "crs_arr_time": ["CRSArrTime"],
        "cancelled": ["Cancelled"],
        "cancellation_code": ["CancellationCode"],
        "diverted": ["Diverted"],
    },
    "flightprices": {
        k: []
        for k in [
            "legId",
            "searchDate",
            "flightDate",
            "startingAirport",
            "destinationAirport",
            "fareBasisCode",
            "travelDuration",
            "elapsedDays",
            "isBasicEconomy",
            "isRefundable",
            "isNonStop",
            "baseFare",
            "totalFare",
            "seatsRemaining",
            "totalTravelDistance",
            "segmentsDepartureTimeEpochSeconds",
            "segmentsArrivalAirportCode",
            "segmentsDepartureAirportCode",
            "segmentsAirlineCode",
            "segmentsCabinCode",
            "segmentsDurationInSeconds",
            "segmentsDistance",
        ]
    },
    "noaa": {
        "station": ["STATION"],
        "date": ["DATE"],
        "precipitation": ["PRCP"],
        "snow": ["SNOW"],
        "temperature": ["TAVG"],
    },
}
