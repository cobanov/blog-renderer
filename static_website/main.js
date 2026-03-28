(function () {
  function initHighlighting() {
    document.querySelectorAll("pre code").forEach(function (block) {
      hljs.highlightElement(block);
    });
  }

  function syncHljsTheme(isDark) {
    var light = document.getElementById("hljs-light");
    var dark = document.getElementById("hljs-dark");
    if (light && dark) {
      light.disabled = isDark;
      dark.disabled = !isDark;
    }
  }

  function initTheme() {
    var themeToggle = document.getElementById("theme-toggle");
    if (!themeToggle) return;
    var icon = themeToggle.querySelector("i");

    var saved = localStorage.getItem("theme");
    var isDark = saved === "dark";
    document.documentElement.classList.toggle("dark-theme", isDark);
    document.documentElement.classList.toggle("light-theme", !isDark);
    icon.classList.replace(
      isDark ? "fa-moon" : "fa-sun",
      isDark ? "fa-sun" : "fa-moon"
    );
    syncHljsTheme(isDark);

    themeToggle.addEventListener("click", function () {
      var nowDark = document.documentElement.classList.toggle("dark-theme");
      document.documentElement.classList.toggle("light-theme");
      localStorage.setItem("theme", nowDark ? "dark" : "light");
      icon.classList.replace(
        nowDark ? "fa-moon" : "fa-sun",
        nowDark ? "fa-sun" : "fa-moon"
      );
      syncHljsTheme(nowDark);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initHighlighting();
    initTheme();
  });
})();
