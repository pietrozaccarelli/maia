import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

def extract_text_from_url(url, timeout=10):
    """Extract readable text content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
            element.decompose()

        # Get all text and clean it up
        text = soup.get_text(separator='\n', strip=True)

        # Clean up the text - remove empty lines and very short lines
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 5]

        return '\n'.join(lines)

    except Exception as e:
        return f"Error extracting text: {str(e)}"

def decode_duckduckgo_url(url):
    """Decode DuckDuckGo redirect URLs to get the real destination"""
    if '//duckduckgo.com/l/' in url:
        try:
            # Add https if missing
            if url.startswith('//'):
                url = 'https:' + url

            # Parse the URL and extract the 'uddg' parameter
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            if 'uddg' in query_params:
                real_url = unquote(query_params['uddg'][0])
                return real_url
        except:
            pass

    # If it's already a direct URL or couldn't decode, return as is
    if url.startswith('http'):
        return url
    elif url.startswith('//'):
        return 'https:' + url
    else:
        return url

def google_it(query, num_links=4):
    """Alternative approach using DuckDuckGo which is more bot-friendly"""
    total_text = ""
    print("Googling it...")
    try:
        encoded_query = urllib.parse.quote_plus(query)
        ddg_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(ddg_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # DuckDuckGo uses simpler selectors - get all results
        results = soup.select('a.result__a')

        if results:
            # Limit to requested number of links (default 5)
            links_to_open = min(len(results), num_links)

            for i in range(links_to_open):
                raw_link = results[i].get('href')
                title = results[i].get_text().strip()

                # Decode the real URL
                real_url = decode_duckduckgo_url(raw_link)

                # Extract text content from the URL
                text_content = extract_text_from_url(real_url)
                total_text += text_content + "\n\n"

                # Add a delay between processing links
                time.sleep(1)  # Respectful delay to avoid overwhelming servers

            return total_text
        else:
            print("Could not find search results")
            return total_text

    except Exception as e:
        print(f"Search failed: {e}")
        return total_text

if __name__ == "__main__":
    query = input("Enter your search query: ")
    num_links = input("How many links to open? (default 5): ").strip()

    # Use default of 5 if no input provided
    if not num_links:
        num_links = 5
    else:
        try:
            num_links = int(num_links)
        except ValueError:
            print("Invalid number, using default of 5")
            num_links = 5

    result = google_it(query, num_links)
    print(result)

