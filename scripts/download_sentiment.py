import os
import urllib.request
import csv

def main():
    print("Starting download of Japanese Sentiment Polarity Dictionary...")
    
    # Define file paths and URLs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    output_path = os.path.join(assets_dir, "sentiment_dict.csv")
    
    noun_url = "http://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/pn.csv.m3.120408.trim"
    verb_url = "http://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/wago.121808.pn"
    
    sentiment_dict = {}

    # Helper function to download and parse with different encodings
    def fetch_url(url):
        print(f"Downloading from: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read()
                
            # Try parsing with UTF-8 first, then EUC-JP or Shift-JIS (Tohoku dictionary is usually UTF-8 or EUC-JP)
            for encoding in ['utf-8', 'euc_jp', 'shift-jis', 'cp932']:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Failed to decode the content with common encodings.")
        except Exception as e:
            print(f"Error downloading or decoding: {e}")
            return None

    # 1. Download and parse nominal sentiment dictionary (名詞編)
    noun_content = fetch_url(noun_url)
    if noun_content:
        lines = noun_content.strip().split('\n')
        print(f"Successfully downloaded noun dictionary. Processing {len(lines)} lines...")
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0].strip()
                polarity = parts[1].strip()
                # standardizing polarity: p=positive, n=negative, e=neutral
                if polarity in ['p', 'n', 'e']:
                    sentiment_dict[word] = polarity
    else:
        print("Warning: Noun dictionary could not be processed.")

    # 2. Download and parse verbal/adjectival sentiment dictionary (用言編)
    verb_content = fetch_url(verb_url)
    if verb_content:
        lines = verb_content.strip().split('\n')
        print(f"Successfully downloaded verb/adjective dictionary. Processing {len(lines)} lines...")
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                # Format: polarity \t word (or representation)
                polarity_raw = parts[0].strip()
                word_raw = parts[1].strip()
                
                # Check for "ポジ" or "ネガ" in polarity string
                if "ポジ" in polarity_raw:
                    polarity = "p"
                elif "ネガ" in polarity_raw:
                    polarity = "n"
                else:
                    continue
                
                # Sometimes word_raw contains representations like "うれしい" or "悲しい"
                # Some entries may contain "する" phrases, or brackets, let's clean it up slightly
                # If there are comma separated variants, split them
                words = [w.strip() for w in word_raw.replace('，', ',').split(',') if w.strip()]
                for w in words:
                    # Remove trailing parts if any, e.g. "〜だ" or "〜する" or focus on base form
                    # We can store the exact entry
                    sentiment_dict[w] = polarity
    else:
        print("Warning: Verb/Adjective dictionary could not be processed.")

    # 3. Output to CSV
    if sentiment_dict:
        print(f"Writing {len(sentiment_dict)} unique sentiment terms to {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['word', 'polarity'])
                for word, pol in sorted(sentiment_dict.items()):
                    writer.writerow([word, pol])
            print("Successfully compiled and saved the sentiment dictionary!")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print("Error: Sentiment dictionary is empty, nothing written.")

if __name__ == "__main__":
    main()
