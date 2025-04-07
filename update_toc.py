#!/usr/bin/env python3
"""
Script to automatically update the README.md table of contents based on the blog articles in the repository.
Usage: python update_toc.py
"""

import os
import re
from datetime import datetime

# Define the categories
CATEGORIES = {
    'machine-learning': {
        'emoji': '🧠',
        'title': 'Machine Learning'
    },
    'computer-vision': {
        'emoji': '👁️',
        'title': 'Computer Vision'
    },
    'robotics': {
        'emoji': '🤖',
        'title': 'Robotics'
    },
    'high-performance-computing': {
        'emoji': '⚡',
        'title': 'High-Performance Computing'
    },
    'programming': {
        'emoji': '💻',
        'title': 'Programming'
    }
}

def extract_metadata(file_path):
    """Extract title, date, and description from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract title from the first heading
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"
    
    # Extract publication date
    date_match = re.search(r'\*Published: ([^*]+)\*', content)
    date_str = date_match.group(1) if date_match else None
    
    # Parse the date
    if date_str:
        try:
            date = datetime.strptime(date_str.strip(), '%B %d, %Y')
        except ValueError:
            try:
                date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            except ValueError:
                date = datetime.now()  # Fallback to current date
    else:
        date = datetime.now()
    
    # Extract the first paragraph as a description
    description_match = re.search(r'^## .*?\n\n(.+?)(?=\n\n|\Z)', content, re.DOTALL | re.MULTILINE)
    description = description_match.group(1) if description_match else ""
    description = description.replace('\n', ' ').strip()
    
    # Truncate description if it's too long
    if len(description) > 100:
        description = description[:97] + "..."
    
    return {
        "title": title,
        "date": date,
        "date_str": date.strftime('%Y-%m-%d'),
        "description": description
    }

def update_toc():
    """Update the README.md table of contents based on the articles in the repository."""
    # Scan all article directories
    articles = []
    base_dir = "articles"
    
    for category in os.listdir(base_dir):
        category_dir = os.path.join(base_dir, category)
        if os.path.isdir(category_dir) and category in CATEGORIES:
            for filename in os.listdir(category_dir):
                if filename.endswith('.md'):
                    file_path = os.path.join(category_dir, filename)
                    relative_path = os.path.join(base_dir, category, filename)
                    metadata = extract_metadata(file_path)
                    
                    articles.append({
                        "category": category,
                        "path": relative_path.replace('\\', '/'),  # Ensure forward slashes for Markdown links
                        "metadata": metadata
                    })
    
    # Sort articles by date, most recent first
    articles.sort(key=lambda x: x["metadata"]["date"], reverse=True)
    
    # Create categorized tables
    tables = {}
    for category in CATEGORIES:
        category_articles = [a for a in articles if a["category"] == category]
        if category_articles:
            tables[category] = "### " + CATEGORIES[category]["emoji"] + " " + CATEGORIES[category]["title"] + "\n"
            tables[category] += "| Date | Title | Description |\n"
            tables[category] += "|------|-------|-------------|\n"
            
            for article in sorted(category_articles, key=lambda x: x["metadata"]["date"], reverse=True):
                tables[category] += f"| {article['metadata']['date_str']} | [{article['metadata']['title']}]({article['path']}) | {article['metadata']['description']} |\n"
    
    # Read the current README
    with open("README.md", 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Replace each category's table
    for category, table in tables.items():
        pattern = f"### {CATEGORIES[category]['emoji']} {CATEGORIES[category]['title']}\n\\| Date \\| Title \\| Description \\|[\\s\\S]*?(?=###|$)"
        replacement = table + "\n"
        readme_content = re.sub(pattern, replacement, readme_content)
    
    # Write the updated README
    with open("README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md table of contents updated with {len(articles)} articles.")

if __name__ == "__main__":
    update_toc() 