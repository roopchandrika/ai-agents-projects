# Postmortem: LIMS Outage

On July 14, 2026 the LIMS was unavailable for nine hours. Sample registration stopped and two time-sensitive
assays had to be repeated.

Root cause: certificate expiry on the internal load balancer. The renewal job had been failing silently since
a service-account password change in May.

Actions: certificates now come from an automated ACME issuer with alerts 21 days before expiry, and the
renewal job reports failures to PagerDuty.
