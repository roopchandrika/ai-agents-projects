# Northwind IT Security Policy

All Northwind accounts use single sign-on through Okta with a hardware key. Every employee receives a YubiKey 5C
on their first day; lost keys must be reported to #it-help within four hours.

Laptops are managed with Jamf and encrypted with FileVault. Source code lives only in the company GitHub
organization; copying firmware repositories to personal devices is a terminable offense.

Report phishing with the Report button in Gmail. Production robot controllers sit on an isolated VLAN and can
only be reached through the Teleport bastion with a ticket number.
