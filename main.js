(function () {
  // Cache code blocks for highlighting
  const codeBlocks = [];

  function initHighlighting() {
    document.querySelectorAll("pre code").forEach((block) => {
      codeBlocks.push(block);
      hljs.highlightElement(block);
    });
  }

  function initTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    if (!themeToggle) return;
    const icon = themeToggle.querySelector("i");

    // Apply saved theme or default to light
    const saved = localStorage.getItem("theme");
    const isDark = saved === "dark";
    document.documentElement.classList.toggle("dark-theme", isDark);
    document.documentElement.classList.toggle("light-theme", !isDark);
    icon.classList.replace(
      isDark ? "fa-moon" : "fa-sun",
      isDark ? "fa-sun" : "fa-moon"
    );

    themeToggle.addEventListener("click", () => {
      const nowDark = document.documentElement.classList.toggle("dark-theme");
      document.documentElement.classList.toggle("light-theme");
      localStorage.setItem("theme", nowDark ? "dark" : "light");
      icon.classList.replace(
        nowDark ? "fa-moon" : "fa-sun",
        nowDark ? "fa-sun" : "fa-moon"
      );
      // Re-highlight with new theme
      codeBlocks.forEach((block) => hljs.highlightElement(block));
    });
  }

  function initLanguage() {
    const langSelect = document.getElementById("lang-select");
    if (!langSelect) return;
    if (!document.documentElement.hasAttribute("data-has-turkish")) return;

    // Determine initial language
    const currentPath = window.location.pathname;
    const defaultLang = currentPath.endsWith("-tr.html") ? "tr" : "en";
    const savedLang = localStorage.getItem("lang");
    const initial = savedLang || defaultLang;
    langSelect.value = initial;
    document.documentElement.lang = initial;

    langSelect.addEventListener("change", () => {
      const newLang = langSelect.value;
      document.documentElement.lang = newLang;
      localStorage.setItem("lang", newLang);
      const base = currentPath.replace(/-(en|tr)\.html$/, "");
      window.location.href = `${base}-${newLang}.html`;
    });
  }

  // Initialize after DOM is ready
  document.addEventListener("DOMContentLoaded", () => {
    initHighlighting();
    initTheme();
    initLanguage();
  });
})();
