import markdown
import os

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
# CSS for basic styling to make it look like a report
CSS_STYLE = """
<style>
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 40px auto;
    max-width: 800px;
    font-size: 12pt;
    color: #333;
}
h1, h2, h3 {
    color: #2c3e50;
    margin-top: 30px;
}
h1 { 
    font-size: 28pt; 
    border-bottom: 3px solid #2c3e50; 
    padding-bottom: 10px; 
    text-align: center;
}
h2 { 
    font-size: 20pt; 
    border-bottom: 1px solid #ddd; 
    padding-bottom: 8px; 
}
h3 { font-size: 16pt; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 25px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}
th {
    background-color: #f8f9fa;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 30px auto;
    border: 1px solid #eee;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
code {
    background-color: #f1f1f1;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: "Courier New", Courier, monospace;
}
blockquote {
    border-left: 5px solid #3498db;
    margin: 20px 0;
    padding: 10px 20px;
    color: #555;
    background-color: #f0f7fb;
}
</style>
"""

def generate_html(md_filename):
    md_path = os.path.join(REPORTS_DIR, md_filename)
    if not os.path.exists(md_path):
        print(f"Error: {md_filename} not found.")
        return

    with open(md_path, "r") as f:
        text = f.read()

    # Convert MD to HTML
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Wrap in HTML body
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{md_filename.replace('.md', '').replace('-', ' ').title()}</title>
        {CSS_STYLE}
    </head>
    <body>
        <div class="report-content">
        {html_content}
        </div>
    </body>
    </html>
    """

    output_filename = md_filename.replace(".md", ".html")
    output_path = os.path.join(REPORTS_DIR, output_filename)

    with open(output_path, "w") as f:
        f.write(full_html)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    reports = [
        "experimentation-of-idqn-architectures-and-activations.md",
        "experimentation-of-ppo-architectures-and-activations.md"
    ]
    
    for r in reports:
        generate_html(r)
