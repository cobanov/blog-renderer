# Blog Renderer

A small static-site renderer for a personal blog. Markdown files go in `blog/`,
the generated Cloudflare Pages-ready site goes in `static_website/`.

## Daily flow

Create a draft:

```bash
python render.py new "My Post Title"
```

Edit the generated file in `blog/`. When it is ready, set `draft: false` or
remove the `draft` line.

Build the static site:

```bash
python render.py
```

Deploy `static_website/` with Cloudflare Pages.

## Post format

Use YAML front matter at the top of each Markdown file:

```markdown
---
title: "My Post Title"
date: 2026-04-24
description: "Short description for SEO and previews."
draft: false
---

# My Post Title

Write the post here.
```

Supported metadata:

- `title`: page title. Falls back to the first `# Heading`.
- `date`: publish date in `YYYY-MM-DD`.
- `description`: meta description. Falls back to the first paragraph.
- `slug`: optional output filename override.
- `draft`: set to `true` to skip rendering.

Older posts that start with a plain `YYYY-MM-DD` line still work, so existing
content does not need to be migrated all at once.
