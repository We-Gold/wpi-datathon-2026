"""The metric vocabulary and the catalogue that describes it."""

from __future__ import annotations

import pytest

from health import metrics, units


class TestNaming:
    def test_healthkit_names_keep_their_prefix(self) -> None:
        assert metrics.is_healthkit(metrics.QUANTITY_PREFIX + "StepCount")
        assert metrics.is_healthkit(metrics.CATEGORY_PREFIX + "SleepAnalysis")

    def test_our_own_names_carry_no_prefix(self) -> None:
        # Rule 2. A name says what was measured and nothing about where it came
        # from, because source_type and source_format already say that.
        for name in metrics.NON_HEALTHKIT_METRICS:
            assert not metrics.is_healthkit(name), name

    def test_no_name_carries_a_unit(self) -> None:
        # health.units converts on the way in and drops the recorded unit, so a
        # name like SedentaryMinutes would be a lie once the value is seconds.
        # The unit lives in the unit column and in CATALOGUE.
        for name in metrics.NON_HEALTHKIT_METRICS:
            for word in ("Minutes", "Seconds", "Hours", "Percent", "Grams"):
                assert word not in name, f"{name} names a unit"

    def test_no_name_carries_a_vendor(self) -> None:
        for name in metrics.NON_HEALTHKIT_METRICS:
            for vendor in ("Fitbit", "PMSys", "Apple", "Google"):
                assert vendor not in name, f"{name} names a vendor"


class TestCatalogue:
    def test_every_metric_we_named_is_catalogued(self) -> None:
        missing = metrics.NON_HEALTHKIT_METRICS - set(metrics.CATALOGUE)
        assert not missing

    def test_dropped_metrics_are_not_catalogued(self) -> None:
        # They are read from PMData and thrown away, so nothing should be able
        # to look one up and believe it is available.
        assert not (metrics.DROPPED_PMDATA_METRICS & set(metrics.CATALOGUE))

    @pytest.mark.parametrize("name,info", sorted(metrics.CATALOGUE.items()))
    def test_entry_is_well_formed(self, name: str, info: metrics.MetricInfo) -> None:
        assert info.domain in metrics.DOMAINS, name
        assert info.daily in metrics.DAILY_RULES, name
        if info.unit is not None:
            assert info.unit in units.CANONICAL_UNITS, name

    def test_category_metrics_hold_no_unit(self) -> None:
        # A category row carries a value_str and no number, so there is nothing
        # for a unit to describe.
        for name, info in metrics.CATALOGUE.items():
            if name.startswith(metrics.CATEGORY_PREFIX):
                assert info.unit is None, name
                assert info.daily in ("duration", "count"), name

    def test_quantity_metrics_hold_a_unit(self) -> None:
        for name, info in metrics.CATALOGUE.items():
            if name.startswith(metrics.QUANTITY_PREFIX):
                assert info.unit is not None, name

    def test_subjective_metrics_are_scores(self) -> None:
        for name in metrics.subjective_metrics():
            assert metrics.CATALOGUE[name].unit == "score", name

    def test_a_measured_score_is_not_subjective(self) -> None:
        # SleepScore is a score the watch computed, not an opinion, so it must
        # not be swept up with the self ratings.
        assert metrics.CATALOGUE["SleepScore"].unit == "score"
        assert not metrics.CATALOGUE["SleepScore"].subjective


class TestLookups:
    def test_a_domain_collects_names_that_share_no_prefix(self) -> None:
        # The reason the catalogue exists. These five are all cardiac and no
        # string match would ever group them.
        cardiac = set(metrics.in_domain("cardiac"))
        for bare in ("HeartRate", "RespiratoryRate", "OxygenSaturation", "VO2Max"):
            assert metrics.QUANTITY_PREFIX + bare in cardiac

    def test_every_domain_has_at_least_one_metric(self) -> None:
        for domain in metrics.DOMAINS:
            assert metrics.in_domain(domain), domain

    def test_an_unknown_domain_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown domain"):
            metrics.in_domain("cardio")

    def test_daily_rules_partition_the_catalogue(self) -> None:
        grouped = metrics.by_daily_rule()
        total = sum(len(names) for names in grouped.values())
        assert total == len(metrics.CATALOGUE)
        assert set(grouped) == set(metrics.DAILY_RULES)
