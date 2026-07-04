# Security Policy

hybrid routes LLM queries between a local model and a frontier API — the privacy promise is that queries answered locally never leave your machine. Vulnerability reports get priority attention.

## Reporting a vulnerability

Please **do not open a public issue** for security reports.

- **Preferred:** [GitHub private vulnerability reporting](https://github.com/askalf/hybrid/security/advisories/new) — creates a private advisory visible only to maintainers.
- **Email:** support@askalf.org with `hybrid security` in the subject.

You'll get an acknowledgement within 72 hours. Please include a minimal reproduction where possible.

## Supported versions

Only the latest release receives security fixes; there are no maintenance branches.

## In scope

- Query content (or fragments of it) sent off-machine when the routing decision or configuration says it should stay local
- Prompt content that manipulates the routing decision itself (adversarial forced escalation)
- Frontier API keys or query content leaking into logs, telemetry, or error output
