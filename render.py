import argparse
import html
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from string import Template
from typing import Any

import markdown2
import yaml


BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteConfig:
    base_dir: Path
    template_path: Path
    pages_src: Path
    blog_src: Path
    output_dir: Path
    assets: tuple[Path, ...]
    markdown_extras: tuple[str, ...]
    base_url: str
    title: str
    author: str
    twitter: str

    @property
    def pages_out(self) -> Path:
        return self.output_dir / self.pages_src.name

    @property
    def blog_out(self) -> Path:
        return self.output_dir / self.blog_src.name

    @property
    def index_out(self) -> Path:
        return self.output_dir / "index.html"

    @property
    def blog_index_out(self) -> Path:
        return self.blog_out / "index.html"


@dataclass(frozen=True)
class Document:
    source_path: Path
    slug: str
    title: str
    date: str
    description: str
    body_markdown: str
    draft: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RenderedPage:
    title: str
    date: str
    filename: str
    basename: str
    canonical: str


class HtmlTemplate:
    def __init__(self, path: Path) -> None:
        self.raw = path.read_text(encoding="utf-8")

    def render(self, **values: str) -> str:
        escaped = {
            key: html.escape(value or "", quote=True)
            for key, value in values.items()
            if key != "content" and key != "json_ld" and key != "date_block"
        }
        escaped["content"] = values.get("content", "")
        escaped["json_ld"] = values.get("json_ld", "")
        escaped["date_block"] = values.get("date_block", "")

        rendered = self.raw
        for key, value in escaped.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", value)
        return rendered


def load_config(path: Path = CONFIG_PATH) -> SiteConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    site = data.get("site", {})

    return SiteConfig(
        base_dir=BASE_DIR,
        template_path=BASE_DIR / data["template"],
        pages_src=BASE_DIR / data["pages_dir"],
        blog_src=BASE_DIR / data["blog_dir"],
        output_dir=BASE_DIR / data["output_dir"],
        assets=tuple(BASE_DIR / asset for asset in data.get("assets", [])),
        markdown_extras=tuple(data.get("markdown_extras", [])),
        base_url=site.get("base_url", "").rstrip("/"),
        title=site.get("title", ""),
        author=site.get("author", ""),
        twitter=site.get("twitter", ""),
    )


def public_url(config: SiteConfig, out_path: Path) -> str:
    rel = out_path.relative_to(config.output_dir)
    parts = list(rel.parts)
    name = parts[-1]

    if name == "index.html":
        parts[-1] = ""
        path = "/".join(parts).strip("/")
        return f"{config.base_url}/{path}".rstrip("/") + "/"

    parts[-1] = name.removesuffix(".html")
    return f"{config.base_url}/{'/'.join(parts)}"


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    raw_meta, body = match.groups()
    metadata = yaml.safe_load(raw_meta) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Front matter must be a YAML mapping")
    return metadata, body.lstrip()


def normalize_date(value: Any, fallback_path: Path) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    return datetime.fromtimestamp(fallback_path.stat().st_mtime).strftime("%Y-%m-%d")


def extract_legacy_date(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})$")

    for index, line in enumerate(lines[:5]):
        match = date_pattern.match(line.strip())
        if match:
            date_str = match.group(1)
            datetime.strptime(date_str, "%Y-%m-%d")
            cleaned = lines[:index] + lines[index + 1 :]
            return date_str, "\n".join(cleaned).strip()
    return None, text


def title_from_markdown(text: str, fallback_slug: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback_slug.replace("-", " ").replace("_", " ").title()


def markdown_description(text: str, max_length: int = 160) -> str:
    skip_prefixes = ("#", "!", ">", "---", "- ", "* ", "```")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or any(stripped.startswith(prefix) for prefix in skip_prefixes):
            continue

        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
        clean = re.sub(r"[*_`]", "", clean).strip()
        if not clean:
            continue
        if len(clean) > max_length:
            return clean[: max_length - 3] + "..."
        return clean
    return ""


def read_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(text)
    legacy_date, body = extract_legacy_date(body)

    slug = str(metadata.get("slug") or path.stem)
    title = str(metadata.get("title") or title_from_markdown(body, slug))
    doc_date = normalize_date(metadata.get("date") or legacy_date, path)
    description = str(metadata.get("description") or markdown_description(body))
    draft = bool(metadata.get("draft", False))

    return Document(
        source_path=path,
        slug=slug,
        title=title,
        date=doc_date,
        description=description,
        body_markdown=body,
        draft=draft,
        metadata=metadata,
    )


def blog_post_json_ld(config: SiteConfig, post: Document, canonical: str) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": post.title,
        "description": post.description,
        "datePublished": post.date,
        "url": canonical,
        "author": {
            "@type": "Person",
            "name": config.author,
            "url": config.base_url,
        },
        "publisher": {
            "@type": "Person",
            "name": config.author,
        },
    }
    return json_script(data)


def person_json_ld(config: SiteConfig) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": config.author,
        "url": config.base_url,
        "sameAs": [
            "https://github.com/cobanov",
            "https://twitter.com/mertcobanov",
            "https://linkedin.com/in/mertcobanoglu",
        ],
    }
    return json_script(data)


def json_script(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return f'<script type="application/ld+json">\n{payload}\n</script>'


def date_block(doc_date: str) -> str:
    if not doc_date:
        return ""
    safe_date = html.escape(doc_date, quote=True)
    return f'<time class="post-date" datetime="{safe_date}">{safe_date}</time>'


def ensure_dirs(config: SiteConfig) -> None:
    for directory in (config.output_dir, config.pages_out, config.blog_out):
        directory.mkdir(parents=True, exist_ok=True)


def clean_output(config: SiteConfig) -> None:
    for directory in (config.pages_out, config.blog_out):
        if directory.exists():
            for html_file in directory.glob("*.html"):
                html_file.unlink()
    if config.index_out.exists():
        config.index_out.unlink()
    logger.info("Cleaned old generated HTML")


def copy_assets(config: SiteConfig) -> None:
    for asset in config.assets:
        if not asset.exists():
            logger.warning("Asset not found, skipping: %s", asset)
            continue
        shutil.copy(asset, config.output_dir / asset.name)
    logger.info("Copied configured assets")


def render_documents(
    config: SiteConfig,
    template: HtmlTemplate,
    src_dir: Path,
    out_dir: Path,
    year: str,
) -> list[RenderedPage]:
    pages: list[RenderedPage] = []

    for md_path in sorted(src_dir.glob("*.md")):
        if src_dir == config.pages_src and md_path.name == "home.md":
            continue

        document = read_document(md_path)
        if document.draft:
            logger.info("Skipped draft: %s", md_path)
            continue

        out_file = out_dir / f"{document.slug}.html"
        canonical = public_url(config, out_file)
        body = markdown2.markdown(document.body_markdown, extras=config.markdown_extras)
        page_title = f"{document.title} — {config.title}"

        out_file.write_text(
            template.render(
                content=body,
                title=page_title,
                date=document.date,
                date_block=date_block(document.date),
                description=document.description,
                year=year,
                canonical_url=canonical,
                og_type="article",
                og_title=document.title,
                json_ld=blog_post_json_ld(config, document, canonical),
            ),
            encoding="utf-8",
        )

        pages.append(
            RenderedPage(
                title=document.title,
                date=document.date,
                filename=f"{out_dir.name}/{out_file.name}",
                basename=out_file.name,
                canonical=canonical,
            )
        )
        logger.info("Built page: %s", out_file)

    return sorted(pages, key=lambda page: page.date, reverse=True)


def build_sub_index(
    config: SiteConfig,
    template: HtmlTemplate,
    pages: list[RenderedPage],
    index_path: Path,
    section_title: str,
    year: str,
) -> None:
    title = section_title.replace("-", " ").replace("_", " ").title()
    items = [f"<h1>{html.escape(title)}</h1>", '<ul class="post-list">']
    for page in pages:
        items.append(
            Template('<li><a href="$href">$title</a> <small>$date</small></li>').substitute(
                href=html.escape(page.basename, quote=True),
                title=html.escape(page.title),
                date=html.escape(page.date),
            )
        )
    items.append("</ul>")

    canonical = public_url(config, index_path)
    template_values = template.render(
        content="\n".join(items),
        title=f"{title} — {config.title}",
        date="",
        date_block="",
        description=f"{title} by {config.author}",
        year=year,
        canonical_url=canonical,
        og_type="website",
        og_title=title,
        json_ld="",
    )
    index_path.write_text(template_values, encoding="utf-8")
    logger.info("Generated section index: %s", index_path)


def build_homepage(
    config: SiteConfig,
    template: HtmlTemplate,
    blog_pages: list[RenderedPage],
    year: str,
) -> None:
    home_path = config.pages_src / "home.md"
    home_text = home_path.read_text(encoding="utf-8") if home_path.exists() else ""
    home_body = (
        markdown2.markdown(home_text, extras=config.markdown_extras)
        if home_text
        else "<h1>Welcome</h1>"
    )

    segments = [
        '<div class="intro">',
        home_body,
        "</div>",
        '<p class="section-label">Writing</p>',
        '<ul class="post-list">',
    ]
    for page in blog_pages:
        segments.append(
            Template('<li><a href="$href">$title</a> <small>$date</small></li>').substitute(
                href=html.escape(page.filename, quote=True),
                title=html.escape(page.title),
                date=html.escape(page.date),
            )
        )
    segments.append("</ul>")

    page_title = f"{config.author} — AI Engineer"
    config.index_out.write_text(
        template.render(
            content="\n".join(segments),
            title=page_title,
            date="",
            date_block="",
            description=markdown_description(home_text) or f"Personal blog of {config.author}",
            year=year,
            canonical_url=public_url(config, config.index_out),
            og_type="website",
            og_title=page_title,
            json_ld=person_json_ld(config),
        ),
        encoding="utf-8",
    )
    logger.info("Generated homepage: %s", config.index_out)


def generate_sitemap(config: SiteConfig, blog_pages: list[RenderedPage]) -> None:
    urls = [
        {
            "loc": public_url(config, config.index_out),
            "priority": "1.0",
            "changefreq": "weekly",
            "lastmod": None,
        },
        {
            "loc": public_url(config, config.blog_index_out),
            "priority": "0.8",
            "changefreq": "weekly",
            "lastmod": None,
        },
    ]

    for page in blog_pages:
        urls.append(
            {
                "loc": page.canonical,
                "priority": "0.7",
                "changefreq": "monthly",
                "lastmod": page.date,
            }
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for url in urls:
        lines.append("  <url>")
        lines.append(f'    <loc>{html.escape(url["loc"])}</loc>')
        if url["lastmod"]:
            lines.append(f'    <lastmod>{url["lastmod"]}</lastmod>')
        lines.append(f'    <changefreq>{url["changefreq"]}</changefreq>')
        lines.append(f'    <priority>{url["priority"]}</priority>')
        lines.append("  </url>")
    lines.append("</urlset>")

    sitemap_path = config.output_dir / "sitemap.xml"
    sitemap_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated sitemap: %s", sitemap_path)


def generate_robots(config: SiteConfig) -> None:
    robots_path = config.output_dir / "robots.txt"
    robots_path.write_text(
        f"User-agent: *\nAllow: /\nSitemap: {config.base_url}/sitemap.xml\n",
        encoding="utf-8",
    )
    logger.info("Generated robots.txt: %s", robots_path)


def build_site() -> None:
    config = load_config()
    template = HtmlTemplate(config.template_path)
    year = str(datetime.now().year)

    ensure_dirs(config)
    clean_output(config)
    copy_assets(config)

    render_documents(config, template, config.pages_src, config.pages_out, year)
    blog_pages = render_documents(config, template, config.blog_src, config.blog_out, year)
    build_sub_index(config, template, blog_pages, config.blog_index_out, config.blog_src.name, year)
    build_homepage(config, template, blog_pages, year)
    generate_sitemap(config, blog_pages)
    generate_robots(config)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "new-post"


def create_post(title: str) -> Path:
    config = load_config()
    slug = slugify(title)
    post_path = config.blog_src / f"{slug}.md"
    if post_path.exists():
        raise FileExistsError(f"Post already exists: {post_path}")

    post_path.write_text(
        "\n".join(
            [
                "---",
                f'title: "{title}"',
                f"date: {date.today().isoformat()}",
                "description: \"\"",
                "draft: true",
                "---",
                "",
                f"# {title}",
                "",
                "Start writing here.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Created draft: %s", post_path)
    return post_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and manage the static blog.")
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser("new", help="Create a new Markdown draft")
    new_parser.add_argument("title", help="Post title")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "new":
        create_post(args.title)
        return
    build_site()


if __name__ == "__main__":
    main()
