import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def remove_think_content(text):
    """Remove <think> tags and their content from text."""
    think_pattern = r'<think>(.*?)</think>'
    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    return cleaned_text

def clean_pdf_text(content):
    """Clean PDF text by normalizing whitespace and removing unwanted characters."""
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\x0c', '', content)
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    content = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', content)
    content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
    return content

def AddMessages(left , right):
    return left + right