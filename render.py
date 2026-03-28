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
BLOG_SRC = BASE_DIR / cfg_data["blog_dir"]
OUT_DIR = BASE_DIR / cfg_data["output_dir"]

PAGES_OUT = OUT_DIR / Path(cfg_data["pages_dir"]).name
BLOG_OUT = OUT_DIR / Path(cfg_data["blog_dir"]).name
INDEX_OUT = OUT_DIR / "index.html"
BLOG_INDEX_OUT = BLOG_OUT / "index.html"

ASSETS = [BASE_DIR / f for f in cfg_data.get("assets", [])]
MD_EXTRAS = cfg_data.get("markdown_extras", [])

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------- Template Cache --------
_raw_template = template_path.read_text(encoding="utf-8")

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


def populate_template(content: str, title: str, date: str, description: str = "", year: str = "") -> str:
    """Inject content, title, date, description, and year into the HTML template."""
    # Replace metadata tokens first, content last — prevents post content from
    # accidentally containing {{ title }} / {{ year }} and getting substituted.
    return (
        _raw_template.replace("{{ title }}", title)
        .replace("{{ date }}", date)
        .replace("{{ description }}", description)
        .replace("{{ year }}", year)
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
        # Strip inline markdown: links [text](url) → text, bold/italic, code
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)  # [text](url) → text
        clean = re.sub(r"[*_`]", "", clean)
        clean = clean.strip()
        if not clean:
            continue
        if len(clean) > max_length:
            return clean[:max_length - 3] + "..."
        return clean
    return ""


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
        body = markdown2.markdown(cleaned_text, extras=MD_EXTRAS)
        html = populate_template(body, title, date_str, description, year)

        out_file = out_dir / f"{slug}.html"
        out_file.write_text(html, encoding="utf-8")

        basename = out_file.name
        full_url = f"{section_name}/{basename}"

        pages.append(
            {
                "title": title,
                "date": date_str,
                "filename": full_url,
                "basename": basename,
            }
        )

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
    html = populate_template("\n".join(items), title, "", year=year)
    index_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated sub-index for {section_title}: {index_path}")


def build_index(blog, year: str):
    """Compose the homepage with blog links."""
    home_md = PAGES_SRC / "home.md"
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

    description = extract_description(home_text) if home_text else "Mert Cobanov's personal blog"
    INDEX_OUT.write_text(
        populate_template("\n".join(segments), "Home", "", description, year),
        encoding="utf-8",
    )
    logger.info(f"Generated homepage: {INDEX_OUT}")


def build_site():
    """Orchestrate directories, assets, pages, and indices."""
    ensure_dirs()
    clean_output()
    copy_assets()

    year = str(datetime.now().year)
    build_section(PAGES_SRC, PAGES_OUT, None, "pages", year)
    blog = build_section(BLOG_SRC, BLOG_OUT, BLOG_INDEX_OUT, cfg_data["blog_dir"], year)
    build_index(blog, year)


def build_section(src: Path, out: Path, index: Path, name: str, year: str):
    """Generalized builder for pages or blog sections."""
    pages = generate_pages(src, out, year)
    if index:
        build_sub_index(pages, index, name, year)
    return pages


if __name__ == "__main__":
    build_site()
