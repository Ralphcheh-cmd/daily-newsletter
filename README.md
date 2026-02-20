# Daily MedTech/Biomedical Top Headline Email (GitHub Actions)

This project sends one daily email containing:

- The top biomedical/medtech headline
- A primary link
- Cross-reference details (how many matching mentions/domains were found)
- Additional matching links

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

- `TOP_NEWS_QUERY` (custom search query; defaults to medtech/biomedical terms)

## Schedule

Workflow file:
`.github/workflows/daily-hi-email.yml`

Current setup runs at Beirut 9:00 AM (DST-safe).

Local dry run example:

```bash
DRY_RUN=true SMTP_HOST=x SMTP_PORT=587 SMTP_USERNAME=x SMTP_PASSWORD=x SMTP_FROM=x SMTP_TO=x python send_daily_hi.py
```
