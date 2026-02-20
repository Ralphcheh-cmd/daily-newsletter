# Daily Hi Email (GitHub Actions)

This project sends a daily email with body:

```text
hi
```

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

## Schedule

Workflow file:
`.github/workflows/daily-hi-email.yml`

Current cron:
`0 13 * * *` (13:00 UTC daily)

Change this cron if you want a different time.
