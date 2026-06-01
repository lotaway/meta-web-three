# TODO

Guidelines: Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.

After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/).

---

### [Backend Admin Missing]

The following backend services have been created, but `apps/backstage-admin/` and `apps/digital-twin/` lack corresponding admin and operation pages. Each needs to be added:

- mall-domain (11 services, most missing admin pages)
- ai-domain (4 services)
- factory-domain / mes-service (production management admin)
- blockchain-domain (2 services)
- erp-domain (6 services: finance, HR, invoice, project, report, settlement)
- platform-domain (7 services: commission, customer service, data analysis, media, message, notification, user behavior)
- supply-chain-domain (6 services: inventory alert, inventory, logistics, procurement, supplier, warehouse)

---

### [Pending Features] (evaluated 2026-06-01)

- Risk control management admin (mall-domain/risk-control-service): Add risk control management page
- Recommendation management admin (mall-domain/recommendation-service): Add recommendation management page
