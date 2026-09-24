# Client-side idle timeout: Cloud Run bills for every second a request is in flight and Streamlit's websocket is a long-lived request that an abandoned tab keeps reconnecting, so the instance never scales to zero, and Streamlit has no server-side idle timeout, so activity is watched in the browser — after IDLE_TIMEOUT_SECONDS a countdown appears, and if it lapses the tab is navigated to a static "session closed" page, navigating away being what does the work, since blanking the page would leave the websocket open.
from __future__ import annotations

import json
import os

import streamlit.components.v1 as components

from i18n import is_rtl, t


def _seconds_from_env(name: str, default: int) -> int:
    # Both timings are env-overridable, so the behaviour can be tried out in seconds locally and retuned in production without a code change.
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# Long enough that reading a generated lesson never trips it, short enough that a tab abandoned after a talk stops costing money within the hour.
IDLE_TIMEOUT_SECONDS = _seconds_from_env("DIDASKALOS_IDLE_TIMEOUT_SECONDS", 20 * 60)
# Grace period between the warning appearing and the session closing.
IDLE_WARNING_SECONDS = _seconds_from_env("DIDASKALOS_IDLE_WARNING_SECONDS", 60)
# Streamlit serves ./static/ here when server.enableStaticServing is on.
SESSION_ENDED_PATH = "/app/static/session-ended.html"

# Placeholder for the live countdown: the localized string is formatted once here, but the number changes every second.
_SECONDS_TOKEN = "%SECONDS%"

# DOM id of the injected <script>, so a rerun can replace its predecessor.
_SCRIPT_ID = "didaskalos-idle-watcher"

# The watcher runs in the app document, not the component iframe, so window and document below are the real page.
_WATCHER_JS = """
(function () {
  var IDLE_MS = __IDLE_MS__;
  var WARN_MS = __WARN_MS__;
  var ENDED_URL = __ENDED_URL__;
  var SECONDS_TOKEN = __SECONDS_TOKEN__;
  var TEXT = __TEXT__;
  var PALETTES = __PALETTES__;
  var FALLBACK_THEME = __FALLBACK_THEME__;

  // theme.py stamps the theme on <html>; read it late so the overlay is right
  // even if the theme changed after this ran.
  function activeTheme() {
    var showing = document.documentElement.getAttribute('data-didaskalos-theme');
    return showing === 'light' || showing === 'dark' ? showing : FALLBACK_THEME;
  }

  if (window.__didaskalosIdle) { window.__didaskalosIdle.destroy(); }

  var EVENTS = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'wheel', 'scroll'];
  var overlay = null;
  var counter = null;
  var deadline = 0;
  var ticker = null;

  function hideOverlay() {
    if (overlay) {
      overlay.remove();
      overlay = null;
      counter = null;
    }
  }

  function showOverlay(secondsLeft) {
    if (!overlay) {
      var COLORS = PALETTES[activeTheme()];
      overlay = document.createElement('div');
      overlay.setAttribute('dir', TEXT.dir);
      overlay.style.cssText = [
        'position:fixed', 'inset:0', 'z-index:2147483000',
        'display:flex', 'align-items:center', 'justify-content:center',
        'background:rgba(0,0,0,0.72)',
        'font-family:' + TEXT.font
      ].join(';');

      var card = document.createElement('div');
      card.style.cssText = [
        'background:' + COLORS.card, 'color:' + COLORS.text, 'border-radius:12px',
        'padding:28px 32px', 'max-width:420px', 'text-align:center',
        'box-shadow:0 10px 40px rgba(0,0,0,0.35)', 'line-height:1.6'
      ].join(';');

      var title = document.createElement('h2');
      title.textContent = TEXT.title;
      title.style.cssText = 'margin:0 0 12px;font-size:1.3rem;color:' + COLORS.accent;

      counter = document.createElement('p');
      counter.style.cssText = 'margin:0 0 20px;font-size:1rem';

      var button = document.createElement('button');
      button.textContent = TEXT.stay;
      button.style.cssText = [
        'background:' + COLORS.accent, 'color:' + COLORS.onAccent, 'border:none',
        'border-radius:8px', 'padding:10px 22px', 'font-size:1rem',
        'cursor:pointer', 'font-family:inherit'
      ].join(';');
      button.addEventListener('click', reset);

      card.appendChild(title);
      card.appendChild(counter);
      card.appendChild(button);
      overlay.appendChild(card);
      document.body.appendChild(overlay);
    }
    counter.textContent = TEXT.body.replace(SECONDS_TOKEN, secondsLeft);
  }

  function reset() {
    hideOverlay();
    deadline = Date.now() + IDLE_MS + WARN_MS;
  }

  function tick() {
    // A running script counts as activity: a full-corpus build fires no mouse or
    // key events for minutes at a time. The status widget is in the DOM only
    // while Streamlit is running.
    if (document.querySelector('[data-testid="stStatusWidget"]')) {
      reset();
      return;
    }

    var remaining = deadline - Date.now();
    if (remaining <= 0) {
      destroy();
      window.location.replace(ENDED_URL + '&theme=' + activeTheme());
    } else if (remaining <= WARN_MS) {
      showOverlay(Math.ceil(remaining / 1000));
    }
  }

  function destroy() {
    if (ticker) { window.clearInterval(ticker); ticker = null; }
    EVENTS.forEach(function (name) {
      document.removeEventListener(name, reset, true);
    });
    hideOverlay();
    window.__didaskalosIdle = null;
  }

  EVENTS.forEach(function (name) {
    document.addEventListener(name, reset, true);
  });
  reset();
  ticker = window.setInterval(tick, 1000);
  window.__didaskalosIdle = { destroy: destroy };
})();
"""

# components.html's iframe sandbox grants allow-same-origin but not allow-top-navigation, so a redirect from inside it is silently blocked — the one thing the watcher must do — and the iframe therefore only injects the watcher into the app document, where it runs unsandboxed.
_BOOTSTRAP_JS = """
<script>
(function () {
  var parentWin = window.parent;
  if (!parentWin || !parentWin.document) { return; }
  var doc = parentWin.document;

  // Every interaction re-injects this component, so drop the previous watcher
  // and its <script> to keep two from running side by side.
  if (parentWin.__didaskalosIdle) { parentWin.__didaskalosIdle.destroy(); }
  var previous = doc.getElementById(__SCRIPT_ID__);
  if (previous) { previous.remove(); }

  var script = doc.createElement('script');
  script.id = __SCRIPT_ID__;
  script.textContent = __WATCHER_SOURCE__;
  doc.head.appendChild(script);
})();
</script>
"""

# Persian needs a stack that has the glyphs; the app's RTL CSS pulls a webfont, but the overlay should not wait on a network request.
_FONT_STACK = {
    True: "'Noto Naskh Arabic','B Lotus',Tahoma,sans-serif",
    False: "'Source Sans Pro','Segoe UI',sans-serif",
}

# The overlay is plain DOM, outside Streamlit's stylesheet, so it is told the palettes, which follow [theme.light] / [theme.dark] in .streamlit/config.toml; both are sent and the overlay picks one from the theme stamp.
_OVERLAY_COLORS = {
    "light": {
        "card": "#dad4c8",
        "text": "#1f1c16",
        "accent": "#3a1712",
        "onAccent": "#cacbc4",
    },
    "dark": {
        "card": "#362923",
        "text": "#e7ded0",
        "accent": "#d4a87a",
        "onAccent": "#1e1714",
    },
}


def render_idle_watcher(lang: str, theme: str = "light") -> None:
    # A zero-height component, so it can be called anywhere in the script; called early, it is installed even on the st.stop() paths.
    rtl = is_rtl(lang)
    text = {
        "title": t("idle_title", lang),
        "body": t("idle_body", lang, seconds=_SECONDS_TOKEN),
        "stay": t("idle_stay", lang),
        "dir": "rtl" if rtl else "ltr",
        "font": _FONT_STACK[rtl],
    }
    # The closed-session page is static and cannot read the app's state, so the language rides in the URL and the watcher appends the theme it can see.
    ended_url = f"{SESSION_ENDED_PATH}?lang={lang}"

    watcher = (
        _WATCHER_JS.replace("__IDLE_MS__", str(IDLE_TIMEOUT_SECONDS * 1000))
        .replace("__WARN_MS__", str(IDLE_WARNING_SECONDS * 1000))
        .replace("__ENDED_URL__", json.dumps(ended_url))
        .replace("__SECONDS_TOKEN__", json.dumps(_SECONDS_TOKEN))
        .replace("__TEXT__", json.dumps(text))
        .replace("__PALETTES__", json.dumps(_OVERLAY_COLORS))
        .replace(
            "__FALLBACK_THEME__",
            json.dumps(theme if theme in _OVERLAY_COLORS else "light"),
        )
    )
    bootstrap = _BOOTSTRAP_JS.replace("__SCRIPT_ID__", json.dumps(_SCRIPT_ID)).replace(
        "__WATCHER_SOURCE__", json.dumps(watcher)
    )
    components.html(bootstrap, height=0)
