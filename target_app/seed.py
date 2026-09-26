"""Synthetic seed data for the proxy target. No real people, no real credentials."""

SESSION_TTL_SECONDS = 900

# Test-only teller credentials for the local sample app (mirrored in .env.example).
TELLERS = {"TELLER01": "teller01-local-test"}

MEMBERS = {
    "10042": {
        "name": "DOE, JANE Q", "ssn": "123-45-6789", "since": "03/14/2011",
        "shares": [
            {"suffix": "00", "type": "01", "desc": "SHARE SAVINGS", "bal": 1234.56},
            {"suffix": "10", "type": "20", "desc": "FREE CHECKING", "bal": 842.10},
        ],
    },
    "20077": {
        "name": "ROE, RICHARD", "ssn": "987-65-4321", "since": "07/01/2019",
        "shares": [
            {"suffix": "00", "type": "01", "desc": "SHARE SAVINGS", "bal": 25.00},
            {"suffix": "12", "type": "20", "desc": "FREE CHECKING", "bal": 3120.44},
        ],
    },
    "30001": {
        "name": "SMITH, ALEX", "ssn": "555-12-3456", "since": "01/02/2001", "restricted": True,
        "shares": [{"suffix": "00", "type": "01", "desc": "SHARE SAVINGS", "bal": 10.00}],
    },
}
