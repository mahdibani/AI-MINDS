APP_CSS = """
:root {
  --bg: #f6f8fc;
  --card: #ffffff;
  --ink: #12213a;
  --muted: #4a5c7a;
  --line: #dbe3f0;
  --accent: #0c7a43;
  --accent-2: #0255aa;
}

body, .gradio-container {
  background: radial-gradient(circle at 20% 10%, #e3efff 0%, var(--bg) 45%) !important;
  color: var(--ink) !important;
}

.block, .gr-box {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: var(--card) !important;
}

.gradio-container h1 {
  color: var(--ink) !important;
}

button.primary {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
}
"""

