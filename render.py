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
PROJ_SRC = BASE_DIR / cfg_data["projects_dir"]
OUT_DIR = BASE_DIR / cfg_data["output_dir"]

PAGES_OUT = OUT_DIR / Path(cfg_data["pages_dir"]).name
BLOG_OUT = OUT_DIR / Path(cfg_data["blog_dir"]).name
PROJ_OUT = OUT_DIR / Path(cfg_data["projects_dir"]).name
INDEX_OUT = OUT_DIR / "index.html"
BLOG_INDEX_OUT = BLOG_OUT / "index.html"
PROJ_INDEX_OUT = PROJ_OUT / "index.html"

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
    for d in (PAGES_OUT, BLOG_OUT, PROJ_OUT):
        if d.exists():
            for f in d.glob("*.html"):
                f.unlink()
    if INDEX_OUT.exists():
        INDEX_OUT.unlink()
    logger.info("Cleaned old output files")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (OUT_DIR, PAGES_OUT, BLOG_OUT, PROJ_OUT):
        d.mkdir(parents=True, exist_ok=True)


def copy_assets():
    """Copy static assets to the output folder."""
    for asset in ASSETS:
        shutil.copy(asset, OUT_DIR / asset.name)
    logger.info(f"Assets copied to {OUT_DIR}")


def populate_template(content: str, title: str, date: str, description: str = "") -> str:
    """Inject content, title, date, description, and year into the HTML template."""
    return (
        _raw_template.replace("{{ content }}", content)
        .replace("{{ title }}", title)
        .replace("{{ date }}", date)
        .replace("{{ description }}", description)
        .replace("{{ year }}", str(datetime.now().year))
    )


def extract_date_from_content(text: str, fallback_path: Path) -> tuple[str, str]:
    """Extract date from markdown content in YYYY-MM-DD format and return cleaned content."""
    lines = text.splitlines()

    # Look for YYYY-MM-DD pattern in the first few lines
    date_pattern = r"^\s*(\d{4}-\d{2}-\d{2})\s*$"

    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        match = re.match(date_pattern, line.strip())
        if match:
            date_str = match.group(1)
            try:
                # Validate the date format
                datetime.strptime(date_str, "%Y-%m-%d")
                # Remove the date line from content
                cleaned_lines = lines[:i] + lines[i + 1 :]
                cleaned_content = "\n".join(cleaned_lines).strip()
                return date_str, cleaned_content
            except ValueError:
                # If parsing fails, continue looking
                continue

    # Fallback to file modification time, return original content
    fallback_date = datetime.fromtimestamp(fallback_path.stat().st_mtime).strftime(
        "%Y-%m-%d"
    )
    return fallback_date, text


def extract_description(text: str, max_length: int = 160) -> str:
    """Extract a plain-text description from markdown content for meta tags."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        clean = re.sub(r"[*_`\[\]()]", "", stripped)
        if len(clean) > max_length:
            return clean[: max_length - 3] + "..."
        return clean
    return ""


def generate_pages(src_dir: Path, out_dir: Path):
    """Build HTML pages from markdown in src_dir, returning metadata."""
    pages = []
    section_name = out_dir.name  # e.g., 'blog'

    for md_path in src_dir.glob("*.md"):
        slug = md_path.stem
        text = md_path.read_text(encoding="utf-8")

        date_str, cleaned_text = extract_date_from_content(text, md_path)

        title = None
        for line in cleaned_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                break
        if not title:
            title = slug.replace("-", " ").replace("_", " ").title()

        description = extract_description(cleaned_text)
        body = markdown2.markdown(cleaned_text, extras=MD_EXTRAS)
        html = populate_template(body, title, date_str, description)

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


def build_sub_index(pages, index_path: Path, section_title: str):
    """Generate a sub-index in its folder, linking by basename."""
    items = [f"<h1>{section_title.title()}</h1>", '<ul class="post-list">']

    for p in pages:
        items.append(
            f'<li><a href="{p["basename"]}">{p["title"]}</a> <small>({p["date"]})</small></li>'
        )

    items.append("</ul>")
    html = populate_template("\n".join(items), section_title.title(), "")
    index_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated sub-index for {section_title}: {index_path}")


def build_index(pages, blog, projects):
    """Compose the homepage with links using full URL."""
    home_md = PAGES_SRC / "home.md"
    home_text = home_md.read_text(encoding="utf-8") if home_md.exists() else ""
    home_body = markdown2.markdown(home_text, extras=MD_EXTRAS) if home_text else "<h1>Welcome</h1>"

    segments = [home_body, "<h2>Blog Posts</h2>", '<ul class="post-list">']
    for p in blog:
        segments.append(
            f'<li><a href="{p["filename"]}">{p["title"]}</a> <small>({p["date"]})</small></li>'
        )
    segments.append("</ul>")

    description = extract_description(home_text) if home_text else "Mert Cobanov's personal blog"
    INDEX_OUT.write_text(
        populate_template("\n".join(segments), "Home", "", description),
        encoding="utf-8",
    )
    logger.info(f"Generated homepage: {INDEX_OUT}")


def build_section(src: Path, out: Path, index: Path, name: str):
    """Generalized builder for pages, blog, or projects sections."""
    pages = generate_pages(src, out)
    if index:
        build_sub_index(pages, index, name)
    return pages


def build_site():
    """Orchestrate directories, assets, pages, and indices."""
    ensure_dirs()
    clean_output()
    copy_assets()

    pages = build_section(PAGES_SRC, PAGES_OUT, None, "pages")
    blog = build_section(BLOG_SRC, BLOG_OUT, BLOG_INDEX_OUT, cfg_data["blog_dir"])
    projects = build_section(
        PROJ_SRC, PROJ_OUT, PROJ_INDEX_OUT, cfg_data["projects_dir"]
    )

    build_index(pages, blog, projects)


if __name__ == "__main__":
    build_site()
