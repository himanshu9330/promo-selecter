"""Interactive Streamlit interface entry point for Promo Finder.

Delegates directly to app.py's run_streamlit() to ensure a single, consistent,
streamlined user experience.
"""

from app import run_streamlit

if __name__ == "__main__" or "streamlit" in __name__:
    run_streamlit()
else:
    run_streamlit()
