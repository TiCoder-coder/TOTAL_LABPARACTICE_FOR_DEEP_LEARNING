"""Presentation filter for Phase 49–59 result dashboards."""
from functools import wraps
import re
from IPython.display import HTML


def results_only(renderer):
    """Omit prose paragraphs, figure captions and header subtitles."""
    @wraps(renderer)
    def render(*args, **kwargs):
        output = renderer(*args, **kwargs)
        html = output.data
        html = re.sub(r'<p\b[^>]*>.*?</p>', '', html, flags=re.S)
        html = re.sub(
            r'<div\b[^>]*class="(?:fcap|cw-d-meta)"[^>]*>.*?</div>',
            '', html, flags=re.S,
        )
        return HTML(html)
    return render
