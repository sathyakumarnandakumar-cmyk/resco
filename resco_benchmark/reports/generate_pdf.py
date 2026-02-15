import markdown
from weasyprint import HTML, CSS
import os

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
# CSS for basic styling
CSS_STYLE = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 40px;
    font-size: 12pt;
}
h1, h2, h3 {
    color: #333;
    margin-top: 20px;
}
h1 { font-size: 24pt; border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { font-size: 18pt; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
h3 { font-size: 14pt; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
}
img {
    max-width: 100%;
    height: auto;
    margin: 20px 0;
    border: 1px solid #ddd;
}
code {
    background-color: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: Consolas, monospace;
}
blockquote {
    border-left: 4px solid #ccc;
    margin: 1.5em 10px;
    padding: 0.5em 10px;
    color: #555;
    background-color: #f9f9f9;
}
"""

def generate_pdf(md_filename):
    md_path = os.path.join(REPORTS_DIR, md_filename)
    if not os.path.exists(md_path):
        print(f"Error: {md_filename} not found.")
        return

    with open(md_path, "r") as f:
        text = f.read()

    # Convert MD to HTML
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Fix image paths: WeasyPrint needs absolute paths usually, or correct relative
    # The current working directory matters. 
    # Our MD files use "plots/image.png". Since we run this script from reports/, it *should* work if base_url is set.
    
    # Wrap in HTML body
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    output_filename = md_filename.replace(".md", ".pdf")
    output_path = os.path.join(REPORTS_DIR, output_filename)

    print(f"Generating {output_filename}...")
    try:
        HTML(string=full_html, base_url=REPORTS_DIR).write_pdf(
            output_path, 
            stylesheets=[CSS(string=CSS_STYLE)]
        )
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    reports = [
        "experimentation-of-idqn-architectures-and-activations.md",
        "experimentation-of-ppo-architectures-and-activations.md"
    ]
    
    for r in reports:
        generate_pdf(r)
