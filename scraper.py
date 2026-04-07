import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_fragrance(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    name = soup.find("h1", {"itemprop": "name"})
    name = name.text.strip() if name else "Unknown"
    
    notes = []
    note_items = soup.find_all("span", {"class": "noteItem"})
    for item in note_items:
        notes.append(item.text.strip())
    
    print(f"Name: {name}")
    print(f"Notes: {notes}")

url = "https://www.fragrantica.com/perfume/Dior/Sauvage-33063.html"
scrape_fragrance(url)
