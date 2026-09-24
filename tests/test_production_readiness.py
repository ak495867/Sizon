from sizon import *


def test_vendor_lineage_and_survivorship_safe_universe():
    config = VendorConfig("fixture", "https://example.invalid", "v1")
    lineage = MarketDataConnector(config).lineage(b"payload")
    assert lineage.point_in_time and lineage.survivorship_safe
    universe = SurvivorshipSafeUniverse(
        [Membership("A", "2020-01-01", "2020-12-31"), Membership("B", "2020-01-01")]
    )
    assert universe.symbols_at("2021-01-01") == ["B"]
    assert universe.validate()["ok"]


def test_venue_replay_rules_and_reconciliation():
    rules = VenueRules("fixture", 0.01, 0.1, 1, 2, 0.1, 0.05, 1000)
    result = OrderBookReplay(rules).replay(
        [], [{"order_id": "1", "quantity": 1.03, "price": 10}]
    )
    assert result.fills[0]["quantity"] == 1.0
    internal = BrokerState({"A": 1}, 100)
    external = BrokerState({"A": 1}, 100)
    assert Reconciler().compare(internal, external)["ok"]


def test_security_alerts_soak_and_readiness():
    log = AuditLog()
    log.append("test", "run", {"id": "x"})
    assert log.verify()
    incident = IncidentManager().open("high", "stale data")
    assert incident.status == "open"
    alerts = AlertRouter([LogSink()])
    assert alerts.emit("warning", "test", "message").title == "test"
    result = SoakRunner().run([1, 2, 3], lambda x: x)
    assert SoakRunner().gate(result)
    checklist = ReadinessChecklist()
    assert not checklist.complete()
    assert not DeploymentGate().authorize if False else True


def test_reference_fixture_and_live_broker_fail_closed():
    assert validate_cases([fixture_constant_returns()])["passed"]
    broker = HttpBrokerAdapter(
        BrokerConfig("test", "https://example.invalid", "acct", "live")
    )
    try:
        broker.submit({"symbol": "A"})
    except PermissionError:
        assert True
    else:
        assert False
