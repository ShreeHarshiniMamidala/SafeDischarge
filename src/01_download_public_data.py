import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from utils import RAW_DIR

SOURCES = {
    "medlineplus_easy_to_read": "https://medlineplus.gov/all_easytoread.html",
    "medlineplus_health_topics": "https://medlineplus.gov/healthtopics.html",
    "medlineplus_xml_page": "https://medlineplus.gov/xml.html",
    "ahrq_ideal_discharge": "https://www.ahrq.gov/patient-safety/patients-families/engagingfamilies/strategy4/index.html",
    "ahrq_ideal_discharge_pdf_info": "https://quality.allianthealth.org/wp-content/uploads/2023/01/AHRQ-Ideal-Discharge-FINAL_508.pdf",

    "heart_failure_discharge": "https://medlineplus.gov/ency/patientinstructions/000114.htm",
    "heart_failure_home_monitoring": "https://medlineplus.gov/ency/patientinstructions/000113.htm",
    "heart_failure_fluids_diuretics": "https://medlineplus.gov/ency/patientinstructions/000112.htm",

    "pneumonia_adults_discharge": "https://medlineplus.gov/ency/patientinstructions/000017.htm",

    "copd_adults_discharge": "https://medlineplus.gov/ency/patientinstructions/000009.htm",
    "copd_quick_relief_drugs": "https://medlineplus.gov/ency/patientinstructions/000026.htm",
    "copd_control_drugs": "https://medlineplus.gov/ency/patientinstructions/000025.htm",
    "copd_ask_doctor": "https://medlineplus.gov/ency/patientinstructions/000215.htm",

    "stroke_discharge": "https://medlineplus.gov/ency/patientinstructions/000132.htm",

    "spine_surgery_discharge": "https://medlineplus.gov/ency/patientinstructions/000313.htm",
    "pancreatitis_discharge": "https://medlineplus.gov/ency/patientinstructions/000201.htm"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

def fetch_page(url, max_retries=3):
    last_error = None
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=(10, 30))
            response.raise_for_status()
            return response.text, response.status_code
        except Exception as e:
            last_error = e
            sleep_time = random.uniform(1.5, 4.0) * attempt
            print(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(sleep_time)

    raise last_error

def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    return "\n".join(lines)

def main():
    rows = []

    for name, url in SOURCES.items():
        try:
            html, status_code = fetch_page(url)
            text = html_to_text(html)

            out_file = RAW_DIR / f"{name}.txt"
            out_file.write_text(text, encoding="utf-8")

            rows.append({
                "source_name": name,
                "url": url,
                "status_code": status_code,
                "file_path": str(out_file),
                "num_chars": len(text),
                "success": 1,
                "error": ""
            })

            print(f"Saved: {out_file}")
            time.sleep(random.uniform(1.0, 2.5))

        except Exception as e:
            rows.append({
                "source_name": name,
                "url": url,
                "status_code": None,
                "file_path": "",
                "num_chars": 0,
                "success": 0,
                "error": str(e)
            })
            print(f"Failed: {name} -> {e}")

    pd.DataFrame(rows).to_csv(RAW_DIR / "public_source_index.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()