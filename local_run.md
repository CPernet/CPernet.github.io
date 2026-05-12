# Running the site locally

## Prerequisites

- **Ruby** ≥ 2.7 — check with `ruby -v`
- **Bundler** — install once with `gem install bundler`

## First-time setup

```bash
cd path/to/CPernet.github.io
bundle install
```

This reads `Gemfile` and installs Jekyll and all GitHub Pages dependencies into a local `vendor/` folder.

## Start the development server

```bash
bundle exec jekyll serve
```

Then open <http://localhost:4000> in your browser.

The server watches for file changes and rebuilds automatically.
Refresh the browser to see updates (CSS and `_config.yml` changes require a restart).

## Useful options

| Flag | Effect |
|------|--------|
| `--livereload` | Auto-refreshes the browser on rebuild |
| `--drafts` | Includes posts in `_drafts/` |
| `--port 5000` | Use a different port |

Example:

```bash
bundle exec jekyll serve --livereload
```

## Keeping dependencies up to date

```bash
bundle update
```

Run this periodically to stay in sync with the GitHub Pages gem.

## Troubleshooting

- **`bundle: command not found`** — run `gem install bundler` first.
- **Port already in use** — add `--port 4001` (or any free port).
- **Changes to `_config.yml` not reflected** — stop the server (`Ctrl+C`) and restart it.
- **Encoding errors on Windows** — add `RUBYOPT="-E utf-8"` before the command:
  ```powershell
  $env:RUBYOPT="-E utf-8"; bundle exec jekyll serve
  ```
