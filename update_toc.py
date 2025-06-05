#!/usr/bin/env python3
"""
Script to automatically update the README.md table of contents based on the blog articles in the repository.
Usage: python update_toc.py
"""

import os
import re
import sys
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from the first heading
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled"
        
        # Extract publication date
        date_match = re.search(r'\*Published: ([^*]+)\*', content)
        date_str = date_match.group(1) if date_match else None
        
        # Parse the date with improved handling of different formats
        if date_str:
            date_formats = ['%B %d, %Y', '%Y-%m-%d', '%d %B %Y', '%Y/%m/%d']
            
            for date_format in date_formats:
                try:
                    date = datetime.strptime(date_str.strip(), date_format)
                    break
                except ValueError:
                    continue
            else:
                # If no format matched, use current date
                print(f"Warning: Could not parse date '{date_str}' in {file_path}. Using current date.")
                date = datetime.now()
        else:
            date = datetime.now()
        
        # Extract the first paragraph as a description (improved pattern)
        description_match = re.search(r'^## .*?\n\n(.+?)(?=\n\n|\n##|\Z)', content, re.DOTALL | re.MULTILINE)
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
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Return default metadata
        return {
            "title": os.path.basename(file_path),
            "date": datetime.now(),
            "date_str": datetime.now().strftime('%Y-%m-%d'),
            "description": "No description available"
        }

def update_toc():
    """Update the README.md table of contents based on the articles in the repository."""
    # Scan all article directories
    articles = []
    base_dir = "articles"
    
    # Check if articles directory exists
    if not os.path.exists(base_dir):
        print(f"Error: '{base_dir}' directory not found. Please make sure you're running the script from the root of the repository.")
        return False
    
    # Files to ignore
    ignore_files = ['TEMPLATE.md', 'README.md']
    
    for category in os.listdir(base_dir):
        category_dir = os.path.join(base_dir, category)
        if os.path.isdir(category_dir) and category in CATEGORIES:
            for filename in os.listdir(category_dir):
                if filename.endswith('.md') and filename not in ignore_files:
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
            tables[category] = f"### {CATEGORIES[category]['emoji']} {CATEGORIES[category]['title']}\n"
            tables[category] += "| Date | Title | Description |\n"
            tables[category] += "|------|-------|-------------|\n"
            
            for article in sorted(category_articles, key=lambda x: x["metadata"]["date"], reverse=True):
                tables[category] += f"| {article['metadata']['date_str']} | [{article['metadata']['title']}]({article['path']}) | {article['metadata']['description']} |\n"
    
    try:
        # Read the current README
        with open("README.md", 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Replace each category's table
        for category, table in tables.items():
            pattern = f"### {CATEGORIES[category]['emoji']} {CATEGORIES[category]['title']}\n\\| Date \\| Title \\| Description \\|[\\s\\S]*?(?=###|$)"
            if re.search(pattern, readme_content):
                # If the pattern exists, replace it
                replacement = table + "\n"
                readme_content = re.sub(pattern, replacement, readme_content)
            else:
                # If the pattern doesn't exist, append to the end of Featured Articles section
                featured_articles_section = "## 📚 Featured Articles"
                if featured_articles_section in readme_content:
                    insertion_point = readme_content.find(featured_articles_section) + len(featured_articles_section)
                    readme_content = readme_content[:insertion_point] + "\n\n<details open>\n<summary><b>" + CATEGORIES[category]['emoji'] + " " + CATEGORIES[category]['title'] + "</b></summary>\n\n" + table + "\n</details>\n" + readme_content[insertion_point:]
        
        # Write the updated README
        with open("README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"README.md table of contents updated with {len(articles)} articles.")
        return True
    
    except Exception as e:
        print(f"Error updating README.md: {str(e)}")
        return False

if __name__ == "__main__":
    success = update_toc()
    if not success:
        sys.exit(1)  # Exit with error code if failed 