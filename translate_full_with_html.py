#!/usr/bin/env python3
"""
Complete PDF translation with HTML output (same as Colab)
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, 'examples')
sys.path.insert(0, 'examples/backends')

def translate_full_pdf():
    """Translate full PDF and generate HTML (same as Colab notebook)"""

    # Import backend
    from ollama_backend import OllamaBackend
    import fitz

    # Configuration (same as Colab)
    PDF_PATH = os.path.expanduser("~/Desktop/2601.09012v2.pdf")
    ARXIV_ID = "2601.09012v2"
    SOURCE_LANG = "en"
    TARGET_LANG = "zh-TW"

    SECTIONS = {
        "abstract": (1, 1),
        "method": (3, 5),
        "experiments": (7, 9),
    }

    print("="*80)
    print("📄 arXiv Bilingual Reader - Local Translation")
    print("="*80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Backend: Ollama (TranslateGemma)")
    print(f"Sections: {list(SECTIONS.keys())}")
    print(f"Total pages: {sum(end - start + 1 for start, end in SECTIONS.values())}")
    print(f"\n🕐 Started: {datetime.now().strftime('%H:%M:%S')}\n")

    # Load backend
    print("🔄 Loading Ollama backend...")
    backend = OllamaBackend()
    backend.load_model()
    print("✅ Model ready!\n")

    # Translate all pages
    results = []

    for section_name, (start_page, end_page) in SECTIONS.items():
        print(f"📖 Translating {section_name.upper()} (Pages {start_page}-{end_page})...")

        for page_num in range(start_page, end_page + 1):
            # Extract text
            doc = fitz.open(PDF_PATH)
            page = doc[page_num - 1]
            original_text = page.get_text().strip()
            doc.close()

            if not original_text:
                print(f"   ⚠️  Page {page_num}: No text, skipping")
                continue

            print(f"   📄 Page {page_num}: Translating {len(original_text)} chars...", end=" ")

            # Translate
            start_time = time.time()
            result = backend.translate(original_text, SOURCE_LANG, TARGET_LANG)
            time_taken = time.time() - start_time

            translation = result['translation']

            # Verify translation
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in translation)
            status = "✅" if has_chinese and len(translation) > 50 else "❌"

            print(f"{status} {time_taken:.1f}s, {len(translation)} chars")

            results.append({
                'page_num': page_num,
                'section': section_name,
                'original': original_text,
                'translation': translation,
                'time': time_taken
            })

    # Generate HTML
    print(f"\n📝 Generating interactive HTML...")
    html_output = generate_html(results, ARXIV_ID, SOURCE_LANG, TARGET_LANG)

    # Save HTML
    output_filename = f"translation_{ARXIV_ID}_{SOURCE_LANG}-{TARGET_LANG}_ollama.html"
    output_path = os.path.expanduser(f"~/Desktop/{output_filename}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"✅ HTML saved: {output_path}")

    # Summary
    total_time = sum(r['time'] for r in results)
    print(f"\n{'='*80}")
    print("✅ Translation Complete!")
    print("="*80)
    print(f"Total pages: {len(results)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average: {total_time/len(results):.1f}s per page")
    print(f"Output: {output_path}")
    print(f"\n🕐 Completed: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

def generate_html(results, arxiv_id, source_lang, target_lang):
    """Generate interactive HTML with navigation"""
    from datetime import datetime

    pages_html = ""
    page_numbers = [r['page_num'] for r in results]

    for idx, result in enumerate(results):
        display_style = "" if idx == 0 else 'style="display: none;"'

        pages_html += f"""
        <div id="page-{idx}" class="translation-page" {display_style}>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 12px 20px; font-weight: bold;">
                📄 {result['section'].upper()} - Page {result['page_num']}
                <span style="float: right; font-weight: normal;">⏱️ {result['time']:.1f}s</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0;">
                <div style="padding: 20px; border-right: 2px solid #e0e0e0; background: #f8f9fa;">
                    <div style="color: #495057; font-weight: bold; margin-bottom: 10px;
                               font-size: 14px;">原文 (Original)</div>
                    <div style="color: #212529; line-height: 1.6; white-space: pre-wrap;
                               font-family: 'Segoe UI', sans-serif;">{result['original'][:3000]}</div>
                </div>
                <div style="padding: 20px; background: #ffffff;">
                    <div style="color: #495057; font-weight: bold; margin-bottom: 10px;
                               font-size: 14px;">翻譯 (Translation)</div>
                    <div style="color: #212529; line-height: 1.6; white-space: pre-wrap;
                               font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif;">{result['translation']}</div>
                </div>
            </div>
        </div>
        """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>TranslateGemma Translation - arXiv:{arxiv_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 0 20px;
                background: #f5f5f5;
            }}
            h1 {{ color: #667eea; text-align: center; margin-bottom: 40px; }}
            .timestamp {{ text-align: center; color: #666; margin-bottom: 30px; }}
            .navigation {{
                position: sticky;
                top: 0;
                background: white;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .nav-btn {{
                padding: 10px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }}
            .nav-btn:hover:not(:disabled) {{ background: #764ba2; }}
            .nav-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
            .page-info {{ color: #495057; font-weight: bold; font-size: 16px; }}
            .translation-page {{
                margin: 20px 0;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                background: white;
            }}
            .hint {{
                text-align: center;
                color: #666;
                font-size: 14px;
                margin: 20px 0;
                padding: 15px;
                background: #fff3cd;
                border-radius: 4px;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                background: #28a745;
                color: white;
                border-radius: 4px;
                font-size: 12px;
                margin-left: 10px;
            }}
        </style>
        <script>
            let currentPage = 0;
            const totalPages = {len(results)};

            function showPage(pageIndex) {{
                if (pageIndex < 0 || pageIndex >= totalPages) return;
                document.querySelectorAll('.translation-page').forEach(page => {{
                    page.style.display = 'none';
                }});
                document.getElementById('page-' + pageIndex).style.display = 'block';
                currentPage = pageIndex;
                updateNavigation();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}

            function updateNavigation() {{
                document.getElementById('prev-btn').disabled = (currentPage === 0);
                document.getElementById('next-btn').disabled = (currentPage === totalPages - 1);
                const pageNum = {page_numbers}[currentPage];
                document.getElementById('page-info').textContent =
                    `Page ${{pageNum}} (${{currentPage + 1}} of ${{totalPages}})`;
            }}

            function nextPage() {{ showPage(currentPage + 1); }}
            function prevPage() {{ showPage(currentPage - 1); }}

            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowLeft') prevPage();
                else if (e.key === 'ArrowRight') nextPage();
            }});

            window.onload = () => {{ updateNavigation(); }};
        </script>
    </head>
    <body>
        <h1>📄 arXiv Bilingual Translation <span class="badge">Ollama Local</span></h1>
        <div class="timestamp">
            Paper: arXiv:{arxiv_id}<br>
            Translation: {source_lang} → {target_lang}<br>
            Backend: Ollama (TranslateGemma)<br>
            Generated: {timestamp}
        </div>

        <div class="navigation">
            <button id="prev-btn" class="nav-btn" onclick="prevPage()">◀ Previous</button>
            <span id="page-info" class="page-info"></span>
            <button id="next-btn" class="nav-btn" onclick="nextPage()">Next ▶</button>
        </div>

        <div class="hint">
            💡 提示：使用 ← → 方向鍵快速切換頁面
        </div>

        {pages_html}

    </body>
    </html>
    """

    return html

if __name__ == "__main__":
    translate_full_pdf()
