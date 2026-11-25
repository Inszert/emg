"""
Design tokens: spacing, font sizes, radii and colours. Plain constants, no Qt.
"""



# ------------------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# DESIGN TOKENS
# ------------------------------------------------------------------------------
# Everything visual is derived from this block. Changing a number here changes
# it everywhere, which is the whole point: before this, paddings and font sizes
# were typed in at ~200 call sites and drifted apart.
#
# Spacing is an 8px rhythm with a 4px half-step. Nothing in the UI should use a
# gap that is not on this scale.
SP_XS = 4
SP_SM = 8
SP_MD = 12
SP_LG = 16
SP_XL = 24

# Type scale. Also a ratio, not a pile of arbitrary sizes.
FS_XS = 9         # hints, captions under a control
FS_SM = 10        # secondary labels, status lines
FS_MD = 11        # body: field labels, list rows
FS_LG = 12        # group headers
FS_XL = 14        # dialog titles
FS_HUGE = 18      # the one number that matters on a screen
FS_BANNER = 34    # the banner's instruction

# Corner radii.
RAD_SM = 4
RAD_MD = 8
RAD_LG = 10

# Control heights, so buttons in a row are never a pixel apart.
H_CONTROL = 30
H_ACTION = 38

COL_BG = "#0f1115"
COL_PANEL = "#171a21"
COL_PANEL_2 = "#1e222b"
COL_BORDER = "#2a2f3a"
COL_TEXT = "#e6e9ef"
COL_TEXT_DIM = "#9aa3b2"
COL_ACCENT = "#38bdf8"
COL_GOOD = "#22c55e"
COL_WARN = "#f59e0b"
COL_BAD = "#ef4444"
