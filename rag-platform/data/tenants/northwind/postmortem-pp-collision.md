# Postmortem: PalletPilot Dock Collision

Incident on August 3, 2026 at a customer site: a PalletPilot robot running firmware 4.2.1 struck a dock door at
low speed. No injuries; the door frame was damaged.

Root cause: a race condition in the obstacle-fusion module dropped lidar frames when the CPU was saturated by
map updates, so the robot relied only on stale data for 700 ms.

Fixes: firmware 4.2.2 moves map updates to a lower-priority thread and adds a watchdog that stops the robot if
sensor data is older than 200 ms. All customer fleets were updated by August 10.
