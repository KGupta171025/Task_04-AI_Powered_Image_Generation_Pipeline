"""
Custom CSS for Image Generation Pipeline Streamlit Frontend.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }

.hero-banner {
    background: linear-gradient(135deg, #090d16 0%, #1e1b4b 50%, #4338ca 100%);
    padding: 2.2rem 2rem 1.8rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.hero-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -0.02em; }
.hero-sub { color: #c7d2fe; font-size: 1rem; }
.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(165, 180, 252, 0.4);
    color: #e0e7ff;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
}

.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.hw-badge-cuda {
    background: #dcfce7;
    color: #15803d;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
}
.hw-badge-cpu {
    background: #e0e7ff;
    color: #3730a3;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
}
</style>
"""
