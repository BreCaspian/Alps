# Contributing to Tech Research Blog

Thank you for your interest in contributing to this tech research blog! This document provides guidelines and instructions for adding content and making improvements.

## How to Contribute

### Adding New Blog Posts

1. **Choose the Right Category**:
   Place your article in the appropriate category folder under `articles/`:
   - `machine-learning/`
   - `computer-vision/`
   - `robotics/`
   - `high-performance-computing/`
   - `programming/`

2. **Create Your Markdown File**:
   - Use a descriptive filename with hyphens between words (e.g., `transformer-architecture.md`)
   - Include proper metadata at the top (see format below)
   - Include at least one main heading (#) for the title

3. **Article Format**:
   ```markdown
   # Your Article Title

   *Published: Month Day, Year*

   ## Introduction

   Your introduction paragraph here.

   ## Main Content

   Your content organized into sections.

   ...

   ## References

   1. Author, A. (Year). Title of paper. *Journal Name*, Volume(Issue), pages.
   
   ---

   *Tags: tag1, tag2, tag3*
   ```

4. **Code Examples**:
   - Include code in fenced code blocks with language specification
   - Use syntax highlighting (e.g., ```python)
   - Comment your code thoroughly

5. **Images and Assets**:
   - Store images in `assets/images/`
   - Store code samples in `assets/code/`
   - Use relative links in your markdown

### Updating the Table of Contents

After adding your article, update the README.md table of contents by running:

```bash
python update_toc.py
```

This script will automatically scan articles in the repository and update the README.md with links to your new content.

## Style Guidelines

### Writing Style

- Use clear, concise language
- Define technical terms when first used
- Break complex topics into digestible sections
- Use descriptive headings and subheadings
- Include practical examples where possible

### Markdown Formatting

- Use heading levels appropriately: # for title, ## for sections, ### for subsections
- Use bullet points for lists where order doesn't matter
- Use numbered lists for sequential steps or ranked items
- Use *italics* for emphasis or introducing new terms
- Use **bold** for strong emphasis or important points
- Use `inline code` for short code snippets, variable names, or commands

## Review Process

1. Once you've created your article, check it for typos, grammar, and technical accuracy
2. Submit a pull request with your changes
3. The blog maintainers will review your content
4. You may be asked to make revisions
5. Once approved, your content will be merged into the main repository

## Code of Conduct

- Be respectful and inclusive in your language
- Provide constructive feedback
- Acknowledge others' contributions
- Focus on technical content and educational value

Thank you for contributing to the tech research community! 