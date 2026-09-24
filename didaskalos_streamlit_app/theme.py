# Light/dark switch. Streamlit owns the theme: the palettes live in .streamlit/config.toml and the frontend picks one from a value it caches in localStorage, which no server-side API can change mid-session, so the switch writes that cache key from the browser and reloads, the only thing that repaints every widget; hence the choice rides in the URL (?theme=) to survive the reload, and a first visit pins light rather than following the OS.
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from i18n import t

THEMES = ("light", "dark")
# Light regardless of the OS setting.
DEFAULT_THEME = "light"

# The icon names the theme the button switches *to*.
_THEME_ICONS = {"light": "☀", "dark": "☾"}

# Containers for the two logo variants: both are rendered and the stylesheet below shows whichever suits the theme on screen, so the logo is never the invisible ink-on-dark combination.
LOGO_CONTAINER_KEYS = {"light": "logo_light", "dark": "logo_dark"}

# Streamlit internals: the frontend's names for the two palettes, and the version of the key it caches the active one under (v2 as of Streamlit 1.56); if a release renames either, the sync stops matching and the app opens in whatever theme Streamlit picked, and the reload guard keeps it from looping.
_FRONTEND_THEME_NAMES = {"light": "Light", "dark": "Dark"}
_ACTIVE_THEME_KEY_VERSION = 2

_SCRIPT_ID = "didaskalos-theme-sync"
_TOGGLE_SCRIPT_ID = "didaskalos-theme-toggle"

# Streamlit's stylesheet exposes no custom properties, so the toggle is told its colours: the text and accent of the theme it sits in.
_TOGGLE_COLORS = {
    "light": {"ink": "#1f1c16", "hover": "#530707", "wash": "rgba(31, 28, 22, 0.10)"},
    "dark": {"ink": "#e7ded0", "hover": "#d4a87a", "wash": "rgba(231, 222, 208, 0.12)"},
}

_TOGGLE_COLOR_RULES = """
%(scope)s .didaskalos-theme-toggle { color: %(ink)s; }
%(scope)s .didaskalos-theme-toggle:hover,
%(scope)s .didaskalos-theme-toggle:focus-visible {
  background-color: %(wash)s;
  color: %(hover)s;
}"""


def _toggle_color_rules(theme: str) -> str:
    # Toggle colours per stamped theme, plus the run's own theme before stamping.
    scopes = [
        ('html[data-didaskalos-theme="light"]', _TOGGLE_COLORS["light"]),
        ('html[data-didaskalos-theme="dark"]', _TOGGLE_COLORS["dark"]),
        ("html:not([data-didaskalos-theme])", _TOGGLE_COLORS[theme]),
    ]
    return "\n".join(
        _TOGGLE_COLOR_RULES % dict(colors, scope=scope) for scope, colors in scopes
    )

_TOGGLE_CSS = """
<style>
/* Sized to sit beside the ☰ menu and the run indicator in Streamlit's toolbar. */
.didaskalos-theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 0.5rem;
  font-size: 1.05rem;
  line-height: 1;
  text-decoration: none;
  transition: background-color 120ms ease, color 120ms ease;
}
%(theme_rules)s
/* Fallback position for a Streamlit whose toolbar this script cannot find. */
.didaskalos-theme-toggle--floating {
  position: fixed;
  top: 0.6rem;
  inset-inline-end: 5.5rem;
  z-index: 999991;
}

/* Show the logo that suits the theme on screen: the script below stamps <html> from the rendered background, so this holds however the theme was set. */
html[data-didaskalos-theme="light"] .st-key-%(dark_logo)s,
html[data-didaskalos-theme="dark"] .st-key-%(light_logo)s,
html:not([data-didaskalos-theme]) .st-key-%(unstamped_logo)s {
  display: none;
}

/* The logo is a mark, not a figure: no fullscreen hover control. */
[data-testid="stSidebar"] [data-testid="stElementToolbar"],
[data-testid="stSidebar"] [data-testid="stBaseButton-elementToolbar"] {
  display: none !important;
}
</style>
"""

# Keeps one toggle in the toolbar: the link is the app's own node, not a widget, so React never reconciles it, and the observer puts it back if a re-render drops it.
_TOGGLE_JS = """
(function () {
  var SETTINGS = __SETTINGS__;
  var CLASS = 'didaskalos-theme-toggle';
  var target = null;

  if (window.__didaskalosThemeToggle) { window.__didaskalosThemeToggle.destroy(); }

  var link = document.createElement('a');
  link.className = CLASS;
  link.href = '#';

  link.addEventListener('click', function (event) {
    event.preventDefault();
    if (!target) { return; }
    // Pin the cached choice before navigating, so the next page already loads
    // in the new theme and the sync has nothing to do.
    try {
      var key = 'stActiveTheme-' + window.location.pathname + '-v' + SETTINGS.keyVersion;
      window.localStorage.setItem(key, JSON.stringify(SETTINGS[target].cached));
    } catch (e) {}
    var url = new URL(window.location.href);
    url.searchParams.set('theme', target);
    window.location.assign(url.toString());
  });

  // Streamlit exposes no theme flag in the DOM, so read the page's background:
  // it is right however the theme was set. Everything that has to match what is
  // on screen keys off this stamp.
  function stamp() {
    var parts = String(window.getComputedStyle(document.body).backgroundColor)
      .match(/[\\d.]+/g);
    if (!parts || parts.length < 3) { return; }
    var luminance =
      (0.299 * Number(parts[0]) + 0.587 * Number(parts[1]) +
        0.114 * Number(parts[2])) / 255;
    var showing = luminance < 0.5 ? 'dark' : 'light';
    document.documentElement.setAttribute('data-didaskalos-theme', showing);

    var wanted = showing === 'dark' ? 'light' : 'dark';
    if (wanted !== target) {
      target = wanted;
      link.textContent = SETTINGS[target].icon;
      link.title = SETTINGS[target].label;
      link.setAttribute('aria-label', SETTINGS[target].label);
    }
  }

  function place() {
    stamp();
    if (!link.isConnected || !document.body.contains(link)) {
      var toolbar = document.querySelector('[data-testid="stToolbar"]');
      var menu = toolbar && toolbar.querySelector('[data-testid="stMainMenu"]');
      var host = (menu && menu.parentElement) || toolbar;
      if (host) {
        link.classList.remove(CLASS + '--floating');
        host.insertBefore(link, host.firstChild);
      } else {
        link.classList.add(CLASS + '--floating');
        document.body.appendChild(link);
      }
    }
  }

  var pending = false;
  function schedule() {
    if (pending) { return; }
    pending = true;
    window.requestAnimationFrame(function () { pending = false; place(); });
  }

  place();
  // Nodes coming and going put the link back; class changes catch a switch made
  // from Streamlit's ☰ menu, which restyles in place and would otherwise leave
  // the stamp, and the logo with it, describing the old theme.
  var observer = new MutationObserver(schedule);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class']
  });

  window.__didaskalosThemeToggle = {
    destroy: function () {
      observer.disconnect();
      link.remove();
      window.__didaskalosThemeToggle = null;
    }
  };
})();
"""

# Runs in the app document (see idle_timeout for why), and reloads only when the page is showing the wrong theme, so the common case costs nothing.
_SYNC_JS = """
(function () {
  var WANTED = __WANTED__;
  var KEY = 'stActiveTheme-' + window.location.pathname + '-v' + __KEY_VERSION__;

  // Once per page load, not per rerun: afterwards the browser is left alone, so
  // a theme picked from Streamlit's ☰ menu sticks.
  if (window.__didaskalosThemeApplied) { return; }
  window.__didaskalosThemeApplied = true;

  // One reload per requested theme per tab, so a renamed cache key cannot leave
  // the page reloading forever.
  var GUARD = 'didaskalos-theme-reload';

  var cached = null;
  try { cached = JSON.parse(window.localStorage.getItem(KEY)); } catch (e) {}

  // A cached "System", or nothing cached, means the OS decides; resolve that the
  // way the frontend does before judging what is on screen.
  var showing =
    cached === 'Light' || cached === 'Dark'
      ? cached
      : window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'Dark'
        : 'Light';

  try {
    if (cached !== WANTED) { window.localStorage.setItem(KEY, JSON.stringify(WANTED)); }
  } catch (e) {
    return; // No storage, no way to make a reload stick.
  }

  if (showing === WANTED) {
    try { window.sessionStorage.removeItem(GUARD); } catch (e) {}
    return;
  }

  try {
    if (window.sessionStorage.getItem(GUARD) === WANTED) { return; }
    window.sessionStorage.setItem(GUARD, WANTED);
  } catch (e) {
    return;
  }
  window.location.reload();
})();
"""

# components.html renders a sandboxed iframe whose scripts cannot navigate the top-level page, so this injects the sync into the app document instead.
_BOOTSTRAP_JS = """
<script>
(function () {
  var parentWin = window.parent;
  if (!parentWin || !parentWin.document) { return; }
  var doc = parentWin.document;

  var previous = doc.getElementById(__SCRIPT_ID__);
  if (previous) { previous.remove(); }

  var script = doc.createElement('script');
  script.id = __SCRIPT_ID__;
  script.textContent = __SYNC_SOURCE__;
  doc.head.appendChild(script);
})();
</script>
"""


def resolve_theme() -> str:
    # The active theme, seeded into session state from the URL, in the same order as the language (URL -> session state -> default): this is the intent for a page load, not what the browser is showing, and anything that must match the screen (logo, toggle direction, idle overlay) reads the browser instead.
    requested = st.query_params.get("theme")
    if requested in THEMES and requested != st.session_state.get("theme"):
        st.session_state["theme"] = requested
    theme = st.session_state.get("theme", DEFAULT_THEME)
    if st.query_params.get("theme") != theme:
        st.query_params["theme"] = theme
    return theme


def _other_theme(theme: str) -> str:
    return "dark" if theme == "light" else "light"


def render_theme_sync(theme: str) -> None:
    # Make the browser show theme, reloading the page if it does not.
    sync = _SYNC_JS.replace(
        "__WANTED__", json.dumps(_FRONTEND_THEME_NAMES[theme])
    ).replace("__KEY_VERSION__", str(_ACTIVE_THEME_KEY_VERSION))
    bootstrap = _BOOTSTRAP_JS.replace("__SCRIPT_ID__", json.dumps(_SCRIPT_ID)).replace(
        "__SYNC_SOURCE__", json.dumps(sync)
    )
    components.html(bootstrap, height=0)


def render_theme_toggle(lang: str, theme: str) -> None:
    # The toggle goes in Streamlit's top toolbar as a plain link rather than an st.button: switching reloads the page anyway, so a link carrying the new ?theme= does in one navigation what a widget would do in a rerun and a reload, and it is a node the app owns rather than a React-managed one; both directions are sent over and the script picks by what is on screen.
    settings = {
        "keyVersion": _ACTIVE_THEME_KEY_VERSION,
        **{
            variant: {
                "icon": _THEME_ICONS[variant],
                "label": t(f"theme_switch_to_{variant}", lang),
                "cached": _FRONTEND_THEME_NAMES[variant],
            }
            for variant in THEMES
        },
    }
    styles = {
        "theme_rules": _toggle_color_rules(theme),
        "light_logo": LOGO_CONTAINER_KEYS["light"],
        "dark_logo": LOGO_CONTAINER_KEYS["dark"],
        "unstamped_logo": LOGO_CONTAINER_KEYS[_other_theme(theme)],
    }
    st.markdown(_TOGGLE_CSS % styles, unsafe_allow_html=True)
    toggle = _TOGGLE_JS.replace("__SETTINGS__", json.dumps(settings))
    bootstrap = _BOOTSTRAP_JS.replace(
        "__SCRIPT_ID__", json.dumps(_TOGGLE_SCRIPT_ID)
    ).replace("__SYNC_SOURCE__", json.dumps(toggle))
    components.html(bootstrap, height=0)
