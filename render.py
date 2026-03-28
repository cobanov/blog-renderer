import json
import logging
from pathlib import Path
import shutil
import markdown2
import yaml
from datetime import datetime
import re

# -------- Configuration --------
BASE_DIR = Path(__file__).parent
cfg_path = BASE_DIR / "config.yaml"
cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

template_path = BASE_DIR / cfg_data["template"]
PAGES_SRC = BASE_DIR / cfg_data["pages_dir"]
BLOG_SRC  = BASE_DIR / cfg_data["blog_dir"]
OUT_DIR   = BASE_DIR / cfg_data["output_dir"]

PAGES_OUT     = OUT_DIR / Path(cfg_data["pages_dir"]).name
BLOG_OUT      = OUT_DIR / Path(cfg_data["blog_dir"]).name
INDEX_OUT     = OUT_DIR / "index.html"
BLOG_INDEX_OUT = BLOG_OUT / "index.html"

ASSETS     = [BASE_DIR / f for f in cfg_data.get("assets", [])]
MD_EXTRAS  = cfg_data.get("markdown_extras", [])

SITE       = cfg_data.get("site", {})
BASE_URL   = SITE.get("base_url", "").rstrip("/")
AUTHOR     = SITE.get("author", "")
TWITTER    = SITE.get("twitter", "")

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------- Template Cache --------
_raw_template = template_path.read_text(encoding="utf-8")


# -------- URL helpers --------

def make_canonical(out_path: Path) -> str:
    """Convert an output file path to its canonical public URL.

    Cloudflare Pages strips .html extensions, so:
      index.html          → https://blog.cobanov.dev/
      blog/index.html     → https://blog.cobanov.dev/blog/
      blog/agents.html    → https://blog.cobanov.dev/blog/agents
    """
    rel = out_path.relative_to(OUT_DIR)
    parts = list(rel.parts)
    name = parts[-1]

    if name == "index.html":
        parts[-1] = ""
        path = "/".join(parts).strip("/")
        return f"{BASE_URL}/{path}".rstrip("/") + "/"
    else:
        parts[-1] = name.removesuffix(".html")
        path = "/".join(parts)
        return f"{BASE_URL}/{path}"


# -------- Structured data --------

def blog_post_json_ld(title: str, date: str, description: str, canonical: str) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "description": description,
        "datePublished": date,
        "url": canonical,
        "author": {
            "@type": "Person",
            "name": AUTHOR,
            "url": BASE_URL,
        },
        "publisher": {
            "@type": "Person",
            "name": AUTHOR,
        },
    }
    return f'<script type="application/ld+json">\n{json.dumps(data, ensure_ascii=False, indent=2)}\n</script>'


def person_json_ld() -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": AUTHOR,
        "url": BASE_URL,
        "sameAs": [
            "https://github.com/cobanov",
            "https://twitter.com/mertcobanov",
            "https://linkedin.com/in/mertcobanoglu",
        ],
    }
    return f'<script type="application/ld+json">\n{json.dumps(data, ensure_ascii=False, indent=2)}\n</script>'


# -------- Helpers --------

def clean_output():
    """Remove old generated HTML files to prevent orphans."""
    for d in (PAGES_OUT, BLOG_OUT):
        if d.exists():
            for f in d.glob("*.html"):
                f.unlink()
    if INDEX_OUT.exists():
        INDEX_OUT.unlink()
    logger.info("Cleaned old output files")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (OUT_DIR, PAGES_OUT, BLOG_OUT):
        d.mkdir(parents=True, exist_ok=True)


def copy_assets():
    """Copy static assets to the output folder."""
    for asset in ASSETS:
        if not asset.exists():
            logger.warning(f"Asset not found, skipping: {asset}")
            continue
        shutil.copy(asset, OUT_DIR / asset.name)
    logger.info(f"Assets copied to {OUT_DIR}")


def populate_template(
    content: str,
    title: str,
    date: str,
    description: str = "",
    year: str = "",
    canonical_url: str = "",
    og_type: str = "website",
    og_title: str = "",
    json_ld: str = "",
) -> str:
    """Inject all variables into the HTML template."""
    # Metadata replaced before content to prevent accidental substitution
    return (
        _raw_template
        .replace("{{ title }}", title)
        .replace("{{ date }}", date)
        .replace("{{ description }}", description)
        .replace("{{ year }}", year)
        .replace("{{ canonical_url }}", canonical_url)
        .replace("{{ og_type }}", og_type)
        .replace("{{ og_title }}", og_title or title)
        .replace("{{ json_ld }}", json_ld)
        .replace("{{ content }}", content)
    )


def extract_date_from_content(text: str, fallback_path: Path) -> tuple[str, str]:
    """Extract date from markdown content in YYYY-MM-DD format and return cleaned content."""
    lines = text.splitlines()
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})$")

    for i, line in enumerate(lines[:5]):
        match = date_pattern.match(line.strip())
        if match:
            date_str = match.group(1)
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                cleaned_lines = lines[:i] + lines[i + 1:]
                return date_str, "\n".join(cleaned_lines).strip()
            except ValueError:
                continue

    fallback_date = datetime.fromtimestamp(fallback_path.stat().st_mtime).strftime("%Y-%m-%d")
    return fallback_date, text


def extract_description(text: str, max_length: int = 160) -> str:
    """Extract a plain-text description from markdown content for meta tags."""
    skip_prefixes = ("#", "!", ">", "---", "- ", "* ", "```")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or any(stripped.startswith(p) for p in skip_prefixes):
            continue
        # [text](url) → text, then strip remaining inline markers
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
        clean = re.sub(r"[*_`]", "", clean).strip()
        if not clean:
            continue
        if len(clean) > max_length:
            return clean[:max_length - 3] + "..."
        return clean
    return ""


# -------- Page builders --------

def generate_pages(src_dir: Path, out_dir: Path, year: str):
    """Build HTML pages from markdown in src_dir, returning metadata."""
    pages = []
    section_name = out_dir.name

    for md_path in src_dir.glob("*.md"):
        if src_dir == PAGES_SRC and md_path.name == "home.md":
            continue

        slug = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        date_str, cleaned_text = extract_date_from_content(text, md_path)

        title = None
        for line in cleaned_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
        if not title:
            title = slug.replace("-", " ").replace("_", " ").title()

        description = extract_description(cleaned_text)
        out_file    = out_dir / f"{slug}.html"
        canonical   = make_canonical(out_file)
        page_title  = f"{title} — cobanov"
        json_ld     = blog_post_json_ld(title, date_str, description, canonical)

        body = markdown2.markdown(cleaned_text, extras=MD_EXTRAS)
        html = populate_template(
            content=body,
            title=page_title,
            date=date_str,
            description=description,
            year=year,
            canonical_url=canonical,
            og_type="article",
            og_title=title,
            json_ld=json_ld,
        )
        out_file.write_text(html, encoding="utf-8")

        pages.append({
            "title":    title,
            "date":     date_str,
            "filename": f"{section_name}/{out_file.name}",
            "basename": out_file.name,
            "canonical": canonical,
        })
        logger.info(f"Built page: {out_file}")

    return sorted(pages, key=lambda p: p["date"], reverse=True)


def build_sub_index(pages, index_path: Path, section_title: str, year: str):
    """Generate a sub-index in its folder, linking by basename."""
    title = section_title.replace("-", " ").replace("_", " ").title()
    items = [f"<h1>{title}</h1>", '<ul class="post-list">']
    for p in pages:
        items.append(
            f'<li><a href="{p["basename"]}">{p["title"]}</a> <small>{p["date"]}</small></li>'
        )
    items.append("</ul>")

    canonical   = make_canonical(index_path)
    page_title  = f"{title} — cobanov"

    html = populate_template(
        content="\n".join(items),
        title=page_title,
        date="",
        year=year,
        canonical_url=canonical,
        og_type="website",
        og_title=title,
    )
    index_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated sub-index for {section_title}: {index_path}")


def build_index(blog, year: str):
    """Compose the homepage with blog links."""
    home_md   = PAGES_SRC / "home.md"
    home_text = home_md.read_text(encoding="utf-8") if home_md.exists() else ""
    home_body = markdown2.markdown(home_text, extras=MD_EXTRAS) if home_text else "<h1>Welcome</h1>"

    segments = [
        '<div class="intro">', home_body, '</div>',
        '<p class="section-label">Writing</p>',
        '<ul class="post-list">',
    ]
    for p in blog:
        segments.append(
            f'<li><a href="{p["filename"]}">{p["title"]}</a> <small>{p["date"]}</small></li>'
        )
    segments.append("</ul>")

    description = extract_description(home_text) if home_text else f"Personal blog of {AUTHOR}"
    canonical   = make_canonical(INDEX_OUT)
    page_title  = f"{AUTHOR} — AI Engineer"

    html = populate_template(
        content="\n".join(segments),
        title=page_title,
        date="",
        description=description,
        year=year,
        canonical_url=canonical,
        og_type="website",
        og_title=page_title,
        json_ld=person_json_ld(),
    )
    INDEX_OUT.write_text(html, encoding="utf-8")
    logger.info(f"Generated homepage: {INDEX_OUT}")


# -------- Sitemap & robots --------

def generate_sitemap(blog: list):
    """Write sitemap.xml covering homepage, blog index, and all posts."""
    urls = []

    # Homepage
    urls.append({
        "loc":        make_canonical(INDEX_OUT),
        "priority":   "1.0",
        "changefreq": "weekly",
        "lastmod":    None,
    })

    # Blog index
    urls.append({
        "loc":        make_canonical(BLOG_INDEX_OUT),
        "priority":   "0.8",
        "changefreq": "weekly",
        "lastmod":    None,
    })

    # Individual posts
    for p in blog:
        urls.append({
            "loc":        p["canonical"],
            "priority":   "0.7",
            "changefreq": "monthly",
            "lastmod":    p["date"],
        })

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for u in urls:
        lines.append("  <url>")
        lines.append(f'    <loc>{u["loc"]}</loc>')
        if u["lastmod"]:
            lines.append(f'    <lastmod>{u["lastmod"]}</lastmod>')
        lines.append(f'    <changefreq>{u["changefreq"]}</changefreq>')
        lines.append(f'    <priority>{u["priority"]}</priority>')
        lines.append("  </url>")
    lines.append("</urlset>")

    sitemap_path = OUT_DIR / "sitemap.xml"
    sitemap_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated sitemap: {sitemap_path}")


def generate_robots():
    """Write robots.txt pointing crawlers to the sitemap."""
    content = f"User-agent: *\nAllow: /\nSitemap: {BASE_URL}/sitemap.xml\n"
    robots_path = OUT_DIR / "robots.txt"
    robots_path.write_text(content, encoding="utf-8")
    logger.info(f"Generated robots.txt: {robots_path}")


# -------- Orchestration --------

def build_section(src: Path, out: Path, index: Path, name: str, year: str):
    """Generalized builder for pages or blog sections."""
    pages = generate_pages(src, out, year)
    if index:
        build_sub_index(pages, index, name, year)
    return pages


def build_site():
    """Orchestrate directories, assets, pages, and indices."""
    ensure_dirs()
    clean_output()
    copy_assets()

    year = str(datetime.now().year)
    build_section(PAGES_SRC, PAGES_OUT, None, "pages", year)
    blog = build_section(BLOG_SRC, BLOG_OUT, BLOG_INDEX_OUT, cfg_data["blog_dir"], year)

    build_index(blog, year)
    generate_sitemap(blog)
    generate_robots()


if __name__ == "__main__":
    build_site()
