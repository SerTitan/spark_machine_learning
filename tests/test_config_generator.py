"""Unit-тесты модуля recommender.config_generator."""

import pytest
from recommender.config_generator import config_distribution_warnings, generate_candidates

# Константы совпадают с допустимыми значениями из collect_hibench_data.sh
_VALID_CODECS = {"lz4", "snappy"}
_VALID_INFLIGHT = {48, 56, 64, 72, 80, 88, 96}
_VALID_FILE_BUF = {32, 48, 64, 80, 96, 112, 128}


class TestGenerateCandidates:
    def test_returns_requested_count(self):
        cands = generate_candidates(3, 3, 6, "large", n=50)
        assert len(cands) == 50

    def test_executor_cores_within_topology(self):
        worker_cores = 4
        cands = generate_candidates(2, worker_cores, 8, "large", n=200)
        assert all(c["executor_cores"] <= worker_cores for c in cands)

    def test_executor_cores_min_one(self):
        cands = generate_candidates(3, 3, 6, "large", n=100)
        assert all(c["executor_cores"] >= 1 for c in cands)

    def test_executor_memory_within_topology(self):
        worker_mem = 6
        cands = generate_candidates(3, 3, worker_mem, "large", n=200)
        # executor_memory_mb не превышает min(worker_mem, 8) * 1024
        cap_mb = min(worker_mem, 8) * 1024
        assert all(c["executor_memory_mb"] <= cap_mb for c in cands)

    def test_executor_memory_multiple_of_1g(self):
        cands = generate_candidates(3, 3, 6, "large", n=100)
        assert all(c["executor_memory_mb"] % 1024 == 0 for c in cands)

    def test_executor_instances_within_topology(self):
        workers = 4
        cands = generate_candidates(workers, 4, 8, "large", n=200)
        assert all(c["executor_instances"] <= workers for c in cands)

    def test_profile_propagated(self):
        for profile in ("small", "large"):
            cands = generate_candidates(2, 4, 8, profile, n=10)
            assert all(c["profile"] == profile for c in cands)

    def test_topology_propagated(self):
        cands = generate_candidates(5, 6, 12, "large", n=10)
        assert all(c["topology_workers"] == 5 for c in cands)
        assert all(c["topology_worker_cores"] == 6 for c in cands)
        assert all(c["topology_worker_mem_gb"] == 12 for c in cands)

    def test_codec_valid_values(self):
        cands = generate_candidates(3, 3, 6, "large", n=200)
        assert all(c["io_codec"] in _VALID_CODECS for c in cands)

    def test_maxsizeinflight_valid_values(self):
        cands = generate_candidates(3, 3, 6, "large", n=200)
        assert all(c["maxSizeInFlight_mb"] in _VALID_INFLIGHT for c in cands)

    def test_shuffle_file_buffer_valid_values(self):
        cands = generate_candidates(3, 3, 6, "large", n=200)
        assert all(c["shuffle_file_buffer_kb"] in _VALID_FILE_BUF for c in cands)

    def test_boolean_fields_are_0_or_1(self):
        cands = generate_candidates(3, 3, 6, "large", n=100)
        for field in ("shuffle_compress", "spill_compress", "broadcast_compress", "rdd_compress"):
            assert all(c[field] in (0, 1) for c in cands), f"{field} out of range"

    def test_memory_fraction_valid_range(self):
        cands = generate_candidates(3, 3, 6, "large", n=200)
        assert all(0.3 <= c["memory_fraction"] <= 0.8 for c in cands)
        assert all(0.3 <= c["memory_storageFraction"] <= 0.8 for c in cands)


class TestConstraints:
    def test_max_executor_cores_respected(self):
        limit = 2
        cands = generate_candidates(4, 8, 16, "large", n=200,
                                    constraints={"max_executor_cores": limit})
        assert all(c["executor_cores"] <= limit for c in cands)

    def test_max_executor_memory_respected(self):
        limit_mb = 2048
        cands = generate_candidates(4, 8, 16, "large", n=200,
                                    constraints={"max_executor_memory_mb": limit_mb})
        assert all(c["executor_memory_mb"] <= limit_mb for c in cands)

    def test_max_executor_instances_respected(self):
        limit = 2
        cands = generate_candidates(8, 6, 12, "large", n=200,
                                    constraints={"max_executor_instances": limit})
        assert all(c["executor_instances"] <= limit for c in cands)

    def test_all_constraints_combined(self):
        cands = generate_candidates(8, 8, 16, "large", n=200, constraints={
            "max_executor_cores": 2,
            "max_executor_memory_mb": 3072,
            "max_executor_instances": 3,
        })
        assert all(c["executor_cores"] <= 2 for c in cands)
        assert all(c["executor_memory_mb"] <= 3072 for c in cands)
        assert all(c["executor_instances"] <= 3 for c in cands)

    def test_rng_seed_reproducible(self):
        a = generate_candidates(3, 3, 6, "large", n=20, rng_seed=42)
        b = generate_candidates(3, 3, 6, "large", n=20, rng_seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_candidates(3, 3, 6, "large", n=20, rng_seed=1)
        b = generate_candidates(3, 3, 6, "large", n=20, rng_seed=2)
        assert a != b


class TestDistributionWarnings:
    def test_generated_candidate_has_no_distribution_warnings(self):
        cfg = generate_candidates(3, 3, 6, "large", n=1, rng_seed=42)[0]
        assert config_distribution_warnings(cfg, 3, 3, 6) == []

    def test_non_grid_numeric_value_warns(self):
        cfg = generate_candidates(3, 3, 6, "large", n=1, rng_seed=42)[0]
        cfg["executor_memory_mb"] = 1536
        warnings = config_distribution_warnings(cfg, 3, 3, 6)
        assert any("executor_memory_mb" in w and "outside sampled design grid" in w for w in warnings)

    def test_unknown_codec_warns(self):
        cfg = generate_candidates(3, 3, 6, "large", n=1, rng_seed=42)[0]
        cfg["io_codec"] = "zstd"
        warnings = config_distribution_warnings(cfg, 3, 3, 6)
        assert any("io_codec" in w and "not present" in w for w in warnings)
