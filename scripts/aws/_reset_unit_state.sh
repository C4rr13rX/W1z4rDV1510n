set -u
# Leave the unit reading `inactive (dead)` rather than `failed`. The stop was
# deliberate, and a unit that reads `failed` invites the next operator -- or the
# next watchdog -- to diagnose a crash that never happened. This project has
# already paid for a state that looks like something it is not.
systemctl reset-failed wizard-curriculum-supervisor 2>&1 || true
systemctl show -p ActiveState -p SubState -p NRestarts wizard-curriculum-supervisor
