const yearNode = document.getElementById("year");
const languageButtons = document.querySelectorAll("[data-lang-button]");
const languagePanels = document.querySelectorAll("[data-lang-panel]");
const translatedNodes = document.querySelectorAll("[data-en][data-fa]");
const localizedLinks = document.querySelectorAll("[data-href-en][data-href-fa]");
const menuIcon = document.getElementById("menu-icon");
const navList = document.querySelector("nav ul");
const themeButton = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-icon");
const themeColorMeta = document.querySelector('meta[name="theme-color"]');
const themedImages = document.querySelectorAll("[data-src-light][data-src-dark]");

const THEME_KEY = "didaskalos-theme";
const THEME_COLORS = { light: "#cdc6b6", dark: "#181310" };
// The button says what a click will do, not what the theme currently is.
const THEME_LABELS = {
  en: { light: "Switch to dark theme", dark: "Switch to light theme" },
  fa: { light: "تغییر به نمای تیره", dark: "تغییر به نمای روشن" },
};

let currentLanguage = "en";
let currentTheme =
  document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";

if (yearNode) {
  yearNode.textContent = new Date().getFullYear();
}

function applyTheme(theme) {
  currentTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", currentTheme);

  // The logo is one flat colour, so each theme gets its own file: the gold original on dark, the dark-ink recolour on light.
  for (const image of themedImages) {
    image.src =
      currentTheme === "dark" ? image.dataset.srcDark : image.dataset.srcLight;
  }

  if (themeColorMeta) {
    themeColorMeta.setAttribute("content", THEME_COLORS[currentTheme]);
  }

  if (themeButton && themeIcon) {
    themeIcon.textContent = currentTheme === "dark" ? "☀" : "☾";
    themeButton.setAttribute("aria-pressed", String(currentTheme === "dark"));
    themeButton.setAttribute(
      "aria-label",
      THEME_LABELS[currentLanguage][currentTheme],
    );
    themeButton.setAttribute("title", THEME_LABELS[currentLanguage][currentTheme]);
  }
}

function setLanguage(language) {
  currentLanguage = language === "fa" ? "fa" : "en";

  for (const panel of languagePanels) {
    panel.hidden = panel.dataset.langPanel !== language;
  }

  for (const node of translatedNodes) {
    node.textContent = node.dataset[language];
  }

  // In-page nav targets differ per panel: the hidden panel's sections cannot be scrolled to, so each link points at the section inside the visible one.
  for (const link of localizedLinks) {
    const target = link.dataset[language === "fa" ? "hrefFa" : "hrefEn"];
    if (target) {
      link.setAttribute("href", target);
    }
  }

  for (const button of languageButtons) {
    button.classList.toggle(
      "is-active",
      button.dataset.langButton === language,
    );
  }

  document.documentElement.lang = language;
  document.documentElement.dir = language === "fa" ? "rtl" : "ltr";

  applyTheme(currentTheme); // the theme button's label is language-dependent
}

for (const button of languageButtons) {
  button.addEventListener("click", () => {
    setLanguage(button.dataset.langButton);
  });
}

if (themeButton) {
  themeButton.addEventListener("click", () => {
    const next = currentTheme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch (e) {}
  });
}

if (menuIcon && navList) {
  menuIcon.addEventListener("click", () => {
    navList.classList.toggle("show");
  });
}

setLanguage("en");
