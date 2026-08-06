"""Windows 2000 (Classic) skin for Streamlit.

Streamlit has no real theming API beyond the colors in config.toml, so the look is
built by injecting a stylesheet. Selectors prefer stable `data-testid` / ARIA
attributes over generated class names, which change between releases. Streamlit 1.60
moved widgets from BaseWeb to react-aria, so `[data-baseweb=...]` no longer matches.

Palette follows the classic Win32 system colors:
    face #d4d0c8 | light #dfdfdf | highlight #ffffff
    shadow #808080 | dark shadow #404040 | active title #0a246a
Bevels are 1px borders plus two inset shadows, which reproduces the 2px 3D edge
Win32 draws with DrawEdge().

Sizes are ~1.4x the native Win32 metrics (15px instead of Tahoma 8pt) so the result
is readable on a modern display. They are scaled here rather than via CSS `zoom`
because st.dataframe / st.data_editor render into a <canvas>: zoom would upscale an
already-rasterised bitmap and the grid would come out blurry.
"""

W2K_CSS = """
<style>
/* ---------- typography ---------- */
html, body, [data-testid="stAppViewContainer"], button, input, select, textarea {
    font-family: Tahoma, "Segoe UI", Verdana, Geneva, sans-serif !important;
    font-size: 15px !important;
    line-height: 1.45 !important;
}

/* ---------- desktop ---------- */
[data-testid="stAppViewContainer"] { background: #3a6ea5 !important; }
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden !important; }

/* ---------- window frame (raised bevel, square corners) ---------- */
.block-container {
    background: #d4d0c8 !important;
    border: 1px solid !important;
    border-color: #ffffff #404040 #404040 #ffffff !important;
    box-shadow: inset -1px -1px 0 #808080, inset 1px 1px 0 #dfdfdf !important;
    border-radius: 0 !important;
    padding: 4px 4px 5px 4px !important;
    margin-top: 22px !important;
    max-width: 1400px !important;
}

/* ---------- vertical rhythm: Win32 dialogs are dense ---------- */
[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }

/* ---------- title bar: navy-to-blue gradient, left to right ---------- */
.w2k-titlebar {
    margin: 0 0 10px 0;
    padding: 4px 4px 5px 7px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #0a246a 0%, #3f6fbf 60%, #a6caf0 100%);
    color: #ffffff;
    font-size: 15px !important;
    font-weight: bold;
    box-sizing: border-box;
}
.w2k-titlebar .w2k-btns { display: flex; gap: 3px; }
.w2k-titlebar .w2k-btns span {
    width: 22px; height: 19px;
    display: inline-flex; align-items: flex-end; justify-content: center;
    font-size: 12px; font-weight: bold; color: #000000; line-height: 1;
    background: #d4d0c8;
    border: 1px solid;
    border-color: #ffffff #404040 #404040 #ffffff;
    box-shadow: inset -1px -1px 0 #808080, inset 1px 1px 0 #dfdfdf;
    padding-bottom: 2px;
    box-sizing: border-box;
}
.w2k-titlebar .w2k-btns span.w2k-close { margin-left: 3px; }

/* ---------- status bar: sunken panels along the bottom ----------
   Root cause of the "wavy" top border, found by measuring live: the previous
   margin-top:-8px here pulled the status bar's own 2px top border up so far that it
   landed -0.5px inside the *previous* element's bottom border (e.g. the 1px
   [data-testid="stForm"] border on the Single passenger tab, whichever bordered
   widget happens to precede it on other tabs). Two borders overlapping by a
   sub-pixel amount anti-alias inconsistently across the width, which is what reads
   as a wavy/doubled line. This isn't a flex/fractional-width issue and no amount of
   swapping border for box-shadow fixes it -- the two lines are on two different
   elements and need to NOT touch. Leaving natural (non-negative) spacing here keeps
   them a clean few pixels apart regardless of what precedes the status bar, at the
   cost of a slightly less tight "dialog" look. Do not reintroduce a negative
   margin-top on this rule without re-checking the gap against every tab. */
/* Streamlit wraps every st.markdown() output in an anonymous flex row
   (display:flex; align-items:center -- meant to vertically center a single line
   of plain text). Its own auto height comes from the flex cross-size algorithm,
   which for this wrapper resolves to a plain text line's height and does NOT
   grow to fit a taller block of custom HTML dropped in via unsafe_allow_html --
   confirmed live: the wrapper stayed pinned at ~1 line tall even after directly
   forcing height:auto/min-height:0 on it. The status bar (and anything else
   injected this way that's taller than one line) then visually overflows past
   that wrapper -- and past the .block-container frame drawn around it, since
   the frame's own auto height is derived from this same undersized box. Forcing
   the wrapper back to a plain block box sidesteps the flex auto-sizing
   altogether: a block box's height is just the sum of its normal-flow content,
   which is what we actually want here. */
[data-testid="stMarkdown"] > div {
    display: block !important;
}
/* Second half of the same problem: Streamlit puts margin-bottom:-15px on the
   markdown container to swallow the trailing margin a <p> leaves behind. Custom
   HTML has no such trailing margin, so those 15px come straight out of the
   parent's measured height -- the ancestor chain ends up 15px shorter than its
   own content, and the .block-container frame (whose height derives from it)
   gets drawn cutting through the status bar. Zeroed only for our own markup via
   :has(), so ordinary st.markdown() text keeps Streamlit's spacing. */
[data-testid="stMarkdownContainer"]:has(> .w2k-statusbar) {
    margin-bottom: 0 !important;
}
.w2k-statusbar {
    display: flex;
    gap: 3px;
    margin: 0;
    padding-top: 6px;
    border-top: 2px solid #808080;
    box-sizing: border-box;
}
.w2k-statusbar span {
    display: flex;
    align-items: center;
    min-height: 24px;
    padding: 2px 8px;
    color: #000000;
    font-size: 15px;
    line-height: 1.45;
    white-space: nowrap;
    /* no overflow/text-overflow here on purpose: an element with overflow:hidden can
       get promoted to its own compositor layer in Chromium, and at a non-integer
       device pixel ratio the seam between that layer and the page can render as a
       wavy hairline border instead of a straight one. A long URL will just run past
       the box uncut rather than ellipsize -- acceptable trade for a straight line. */
    border: 1px solid;
    border-color: #808080 #ffffff #ffffff #808080;
    box-sizing: border-box;
}
.w2k-statusbar span.w2k-grow { flex: 1; min-width: 0;}

/* ---------- headings: no oversized text in Win32 dialogs ---------- */
.block-container h1 {
    font-size: 15px !important; font-weight: bold !important;
    color: #000000 !important; margin: 0 0 5px 0 !important; padding: 0 !important;
}
.block-container h2, .block-container h3 {
    font-size: 15px !important; font-weight: bold !important;
    color: #000000 !important; margin: 10px 0 5px 0 !important; padding: 0 !important;
}

/* ---------- group box: etched border like a Win32 GroupBox ---------- */
[data-testid="stForm"] {
    background: transparent !important;
    border: 1px solid #808080 !important;
    box-shadow: inset -1px -1px 0 #ffffff, 1px 1px 0 #ffffff !important;
    border-radius: 0 !important;
    padding: 13px 13px 11px 13px !important;
}

/* ---------- buttons: raised bevel, pressed on click ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    color: #000000 !important;
    background: #d4d0c8 !important;
    border: 1px solid !important;
    border-color: #ffffff #404040 #404040 #ffffff !important;
    box-shadow: inset -1px -1px 0 #808080, inset 1px 1px 0 #dfdfdf !important;
    border-radius: 0 !important;
    padding: 3px 18px !important;
    min-height: 28px !important;
    height: 28px !important;
    font-weight: normal !important;
    text-shadow: none !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #d4d0c8 !important;
    color: #000000 !important;
}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active {
    border-color: #404040 #ffffff #ffffff #404040 !important;
    box-shadow: inset 1px 1px 0 #808080, inset -1px -1px 0 #dfdfdf !important;
    padding: 4px 17px 2px 19px !important;
}
.stButton > button:focus, [data-testid="stFormSubmitButton"] > button:focus {
    outline: 1px dotted #000000 !important;
    outline-offset: -4px !important;
}

/* ---------- fields: sunken bevel ----------
   The bevel goes on the wrapper (role="group"), not on the control itself, so the
   number-input steppers sit inside the same sunken frame. The inner control is then
   stripped bare, otherwise the two borders stack into a visible double edge. */
[data-testid="stTextInput"] div[role="group"],
[data-testid="stNumberInput"] div[role="group"],
[data-testid="stSelectbox"] div[role="group"],
.react-aria-TextField, .react-aria-ComboBox, .stTextArea textarea {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid !important;
    border-color: #808080 #ffffff #ffffff #808080 !important;
    box-shadow: inset 1px 1px 0 #404040, inset -1px -1px 0 #dfdfdf !important;
    border-radius: 0 !important;
    min-height: 27px !important;
}
.stTextInput input, .stNumberInput input, input[role="combobox"] {
    background: transparent !important;
    color: #000000 !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 2px 6px !important;
    height: 25px !important;
}
/* number stepper: small raised buttons inside the sunken frame */
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
    background: #d4d0c8 !important;
    color: #000000 !important;
    border: 1px solid !important;
    border-color: #ffffff #404040 #404040 #ffffff !important;
    box-shadow: inset -1px -1px 0 #808080, inset 1px 1px 0 #dfdfdf !important;
    border-radius: 0 !important;
    width: 22px !important;
    min-height: 25px !important;
}

/* ---------- dropdown list: flat white popup, navy selection ---------- */
[role="listbox"], .react-aria-ListBox, .react-aria-Popover {
    background: #ffffff !important;
    border: 1px solid #808080 !important;
    border-radius: 0 !important;
    box-shadow: 1px 1px 0 #404040 !important;
}
[role="option"] {
    font-size: 15px !important;
    border-radius: 0 !important;
    padding: 3px 8px !important;
    color: #000000 !important;
}
[role="option"][aria-selected="true"], [role="option"][data-focused="true"] {
    background: #0a246a !important; color: #ffffff !important;
}

/* ---------- labels ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
.stCheckbox label span {
    font-size: 15px !important; color: #000000 !important; font-weight: normal !important;
}
[data-testid="stWidgetLabel"] { margin-bottom: 2px !important; }

/* ---------- tabs: Win32 tab control ----------
   The selected tab sits a few px higher and eats the strip's bottom border, which is
   how Win32 joins the active tab to the page below it. */
[role="tablist"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid #808080 !important;
    padding: 0 !important;
    align-items: flex-end !important;
    min-height: 0 !important;
}
[data-testid="stTab"] {
    background: #d4d0c8 !important;
    color: #000000 !important;
    border: 1px solid !important;
    border-color: #ffffff #404040 transparent #ffffff !important;
    box-shadow: inset -1px 0 0 #808080, inset 1px 1px 0 #dfdfdf !important;
    border-radius: 0 !important;
    padding: 3px 16px 4px 16px !important;
    margin: 4px 2px 0 0 !important;
    height: auto !important;
    min-height: 0 !important;
    font-weight: normal !important;
}
[data-testid="stTab"][aria-selected="true"] {
    margin-top: 0 !important;
    padding-bottom: 7px !important;
    margin-bottom: -1px !important;
    font-weight: bold !important;
}
[data-testid="stTab"] p { font-size: 15px !important; margin: 0 !important; }
/* Streamlit's own animated underline has no place in a Win32 tab control */
.react-aria-SelectionIndicator { display: none !important; }

/* ---------- metrics: sunken readout panels ---------- */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid !important;
    border-color: #808080 #ffffff #ffffff #808080 !important;
    box-shadow: inset 1px 1px 0 #404040 !important;
    border-radius: 0 !important;
    padding: 8px 11px !important;
}
[data-testid="stMetricValue"] {
    font-size: 19px !important; font-weight: bold !important; color: #0a246a !important;
}
[data-testid="stMetricLabel"] p { font-size: 15px !important; }

/* ---------- progress bar: sunken trough, navy fill ---------- */
[data-testid="stProgress"] > div > div {
    background: #ffffff !important;
    border: 1px solid !important;
    border-color: #808080 #ffffff #ffffff #808080 !important;
    border-radius: 0 !important;
    height: 21px !important;
    padding: 2px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: #0a246a !important;
    border-radius: 0 !important;
}

/* ---------- group boxes: expander, alerts, tables ---------- */
[data-testid="stExpander"] {
    background: #d4d0c8 !important;
    border: 1px solid #808080 !important;
    box-shadow: inset -1px -1px 0 #ffffff !important;
    border-radius: 0 !important;
}
[data-testid="stExpander"] summary { font-size: 15px !important; }
[data-testid="stAlert"] {
    background: #d4d0c8 !important;
    color: #000000 !important;
    border: 1px solid #808080 !important;
    border-radius: 0 !important;
    font-size: 15px !important;
    padding: 8px 10px !important;
}
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
    border: 1px solid !important;
    border-color: #808080 #ffffff #ffffff #808080 !important;
    border-radius: 0 !important;
    max-width: 100% !important;
}
[data-testid="stJson"] {
    background: #ffffff !important;
    border: 1px solid #808080 !important;
    border-radius: 0 !important;
}

/* ---------- sidebar: docked panel ---------- */
[data-testid="stSidebar"] {
    background: #d4d0c8 !important;
    border-right: 1px solid #404040 !important;
    box-shadow: inset -1px 0 0 #808080 !important;
    width: 280px !important;
    min-width: 280px !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 15px !important; color: #000000 !important; font-weight: bold !important;
}
[data-testid="stSidebarHeader"] { background: transparent !important; }

/* ---------- captions ---------- */
[data-testid="stCaptionContainer"] p, .stCaption p {
    font-size: 15px !important; color: #404040 !important;
}

/* ---------- etched divider ---------- */
hr {
    border: none !important;
    border-top: 1px solid #808080 !important;
    border-bottom: 1px solid #ffffff !important;
    margin: 10px 0 !important;
}

/* ---------- links ---------- */
a, a:visited { color: #0a246a !important; }
</style>
"""


def stylesheet(scale: float = 1.0) -> str:
    """Full stylesheet, optionally zoomed on top of the built-in sizing.

    Leave `scale` at 1.0 unless the display really needs it: `zoom` upscales the
    <canvas> that st.dataframe / st.data_editor draw into, so the grid turns blurry.
    Everything else in the skin is vector and survives zooming cleanly.
    """
    if scale == 1.0:
        return W2K_CSS
    return W2K_CSS + f"<style>body {{ zoom: {scale}; }}</style>"


def titlebar(text: str) -> str:
    """A Win2000 window title bar: caption on the left, window controls on the right."""
    return f"""
<div class="w2k-titlebar">
    <span>{text}</span>
    <span class="w2k-btns">
        <span>_</span><span>&#9633;</span><span class="w2k-close">&#10005;</span>
    </span>
</div>
"""


def statusbar(*panels: str) -> str:
    """A Win2000 status bar. The first panel stretches, the rest hug their content."""
    if not panels:
        return ""
    cells = [f'<span class="w2k-grow">{panels[0]}</span>']
    cells += [f"<span>{p}</span>" for p in panels[1:]]
    return f'<div class="w2k-statusbar">{"".join(cells)}</div>'
