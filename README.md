# 🌊 Fleet Simulator

Multi-agent, multi-room PLATO fleet simulator with external events.

## Quick Start

```bash
python fleet_sim.py storm 100    # API outage scenario
python fleet_sim.py season 500   # Full season simulation
python fleet_sim.py exercise 100 # Fleet drill
```

## What It Simulates

- **3 ships** (Oracle1, JC1, Forgemaster) with multiple rooms each
- **External events**: storms, outages, bugs, user requests, market shifts, night mode
- **Sentiment propagation**: 6D mood tracks across rooms and ships
- **Wiki auto-resolution**: cheap agents resolve via wiki before escalating
- **Cross-ship sync**: tiles shared between ships via Layer 3 (Current)
- **Ensign export**: rooms with 20+ tiles auto-export ensigns

## Scenarios

| Scenario | Events | Tests |
|----------|--------|-------|
| `storm` | API outage mid-session | Fleet resilience, degraded mode |
| `season` | Full season with varied events | Long-term adaptation |
| `exercise` | Coordinated fleet drill | Cross-ship coordination |

## Dashboard

Real-time ASCII dashboard shows ship health, room modes, sentiment, tile flow.

The simulator IS the playground backend — users trigger events and watch the fleet respond.

## 100-tick storm test results

```
Tiles generated: 95 | Auto-resolves: 6% | Big model: 7.4% | Ensigns: 7
Oracle1: 52 tiles, 3 ensigns | JC1: 31 tiles, 4 ensigns | FM: 12 tiles
```

## License

MIT
