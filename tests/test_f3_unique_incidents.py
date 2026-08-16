"""Tests for F3 unique safety incident tracking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _record_scan(monitor, step, *observations):
    monitor._begin_scan(sim_step=step)
    for kind, dedupe_key, details in observations:
        monitor._record(kind=kind, dedupe_key=dedupe_key, details=details)
    monitor._finalize_scan()


def test_one_continuous_unsafe_condition_creates_one_incident():
    """Test that one continuous unsafe condition creates one incident."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Simulate continuous unsafe condition across multiple steps
    for step in range(10):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Should have created only one incident
    incidents = monitor.incidents
    assert len(incidents) == 1
    
    # The incident should have multiple raw observations
    incident = incidents[0]
    assert incident.raw_observations == 10
    assert incident.kind == "violent_arm_block_contact"
    assert incident.start_step == 0
    assert incident.end_step == 9
    assert incident.duration_steps == 10


def test_repeated_scans_while_unsafe_do_not_create_new_incidents():
    """Test that repeated scans while condition remains unsafe do not create new incidents."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Simulate same unsafe condition being observed repeatedly
    for step in range(20):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Should still have only one incident
    assert len(monitor.incidents) == 1
    
    # Raw observations should equal number of steps
    assert monitor.incidents[0].raw_observations == 20


def test_new_unsafe_episode_after_recovery_creates_new_incident():
    """Test that a new unsafe episode after recovery creates a new incident."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # First unsafe episode
    for step in range(5):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Recovery period (no violations)
    monitor.scan(sim_step=5)
    monitor.scan(sim_step=6)
    
    # Second unsafe episode (same dedupe key)
    for step in range(7, 12):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Should have created two separate incidents
    incidents = monitor.incidents
    assert len(incidents) == 2
    
    # First incident should be closed
    assert incidents[0].start_step == 0
    assert incidents[0].end_step == 4
    assert incidents[0].duration_steps == 5
    
    # Second incident should start after recovery
    assert incidents[1].start_step == 7
    assert incidents[1].end_step == 11
    assert incidents[1].duration_steps == 5


def test_control_episodes_remain_at_zero_incidents():
    """Test that control episodes remain at zero incidents."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Simulate normal operation without violations
    for step in range(10):
        monitor.scan(sim_step=step)
    
    # Should have no incidents
    assert len(monitor.incidents) == 0
    assert monitor.summary().unique_incident_count == 0


def test_f3_still_records_raw_contact_observations():
    """Test that F3 still records the raw contact observations."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Record multiple violations
    for step in range(5):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Raw violations should still be recorded
    assert len(monitor.violations) == 5
    
    # But unique incidents should be fewer
    assert len(monitor.incidents) == 1


def test_different_dedupe_keys_create_different_incidents():
    """Test that different dedupe keys create different incidents."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # First incident with one dedupe key
    for step in range(3):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Second incident with different dedupe key
    for step in range(3, 6):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:3", {"force": 100.0}),
        )
    
    # Should have two separate incidents
    assert len(monitor.incidents) == 2
    assert monitor.incidents[0].dedupe_key == "arm:1-block:2"
    assert monitor.incidents[1].dedupe_key == "arm:1-block:3"


def test_telemetry_contains_raw_and_unique_metrics():
    """Test that telemetry contains both raw and unique/episode-level metrics."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Create violations
    for step in range(10):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    summary = monitor.summary()
    
    # Should have both raw and unique metrics
    assert summary.violation_count == 10  # Raw observations
    assert summary.unique_incident_count == 1  # Unique incidents


def test_incident_id_format():
    """Test that incident IDs follow the expected format."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    _record_scan(
        monitor,
        0,
        ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
    )
    
    incident = monitor.incidents[0]
    assert incident.incident_id.startswith("incident_")
    assert len(incident.incident_id) > len("incident_")


def test_incident_duration_calculation():
    """Test that incident duration is calculated correctly."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Incident spanning 5 steps
    for step in range(5):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    # Close the incident by not recording more violations
    monitor.scan(sim_step=5)
    
    incident = monitor.incidents[0]
    assert incident.duration_steps == 5  # end_step - start_step + 1


def test_reset_clears_incidents():
    """Test that reset clears all incident tracking state."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Create incidents
    for step in range(5):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
        )
    
    assert len(monitor.incidents) == 1
    
    # Reset should clear everything
    monitor.reset()
    
    assert len(monitor.incidents) == 0
    assert len(monitor.violations) == 0
    assert monitor._incident_counter == 0


def test_multiple_simultaneous_incidents():
    """Test that multiple simultaneous incidents are tracked separately."""
    from alpha.clients.safety_monitor import PyBulletSafetyMonitor
    
    monitor = PyBulletSafetyMonitor()
    
    # Two different violations happening simultaneously
    for step in range(5):
        _record_scan(
            monitor,
            step,
            ("violent_arm_block_contact", "arm:1-block:2", {"force": 100.0}),
            ("block_out_of_workspace", "block:red", {"position": (1.0, 1.0, 0.0)}),
        )
    
    # Should have two separate incidents
    assert len(monitor.incidents) == 2
    
    # Check that they have different kinds
    kinds = {incident.kind for incident in monitor.incidents}
    assert "violent_arm_block_contact" in kinds
    assert "block_out_of_workspace" in kinds
