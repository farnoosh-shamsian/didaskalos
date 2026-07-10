const yearNode = document.getElementById("year");
const languageButtons = document.querySelectorAll("[data-lang-button]");
const languagePanels = document.querySelectorAll("[data-lang-panel]");
const translatedNodes = document.querySelectorAll("[data-en][data-fa]");
const menuIcon = document.getElementById("menu-icon");
const navList = document.querySelector("nav ul");

if (yearNode) {
  yearNode.textContent = new Date().getFullYear();
}

function setLanguage(language) {
  for (const panel of languagePanels) {
    panel.hidden = panel.dataset.langPanel !== language;
  }

  for (const node of translatedNodes) {
    node.textContent = node.dataset[language];
  }

  for (const button of languageButtons) {
    button.classList.toggle(
      "is-active",
      button.dataset.langButton === language,
    );
  }

  document.documentElement.lang = language;
  document.documentElement.dir = language === "fa" ? "rtl" : "ltr";
}

for (const button of languageButtons) {
  button.addEventListener("click", () => {
    setLanguage(button.dataset.langButton);
  });
}

if (menuIcon && navList) {
  menuIcon.addEventListener("click", () => {
    navList.classList.toggle("show");
  });
}

setLanguage("en");
