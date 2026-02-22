# Daily MedTech/Biomedical Top Headline Email (GitHub Actions)

This project sends one daily email containing the top 3-4 biomedical/medtech headlines from 5 selected outlets:

- MedTech Dive
- MassDevice
- Fierce Biotech
- STAT
- MedCity News

It runs on GitHub Actions cloud (your PC can be off).

## Required GitHub Secrets

Set these in:
`Repo -> Settings -> Secrets and variables -> Actions`

- `SMTP_HOST` (example: `smtp.gmail.com`)
- `SMTP_PORT` (example: `587`)
- `SMTP_USERNAME` (your SMTP login)
- `SMTP_PASSWORD` (for Gmail: App Password)
- `SMTP_FROM` (sender email)
- `SMTP_TO` (recipient email)
- `SMTP_SECURITY` (`starttls` or `ssl`)

Optional:

- `HEADLINE_COUNT` (3 or 4, defaults to 4)

## Email Format

Each headline is formatted as:

- `I - ...`
- Company research (separate lookup)
- Product research in easy terms + engineering explanation (only if product is mentioned)
- Link / date of release

## Schedule

Workflow file:
`.github/workflows/daily-hi-email.yml`

Current setup runs at Beirut 9:00 AM (DST-safe).

Local dry run example:

```bash
DRY_RUN=true SMTP_HOST=x SMTP_PORT=587 SMTP_USERNAME=x SMTP_PASSWORD=x SMTP_FROM=x SMTP_TO=x python send_daily_hi.py
```
