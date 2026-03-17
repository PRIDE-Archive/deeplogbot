#!/usr/bin/env python3
"""Build geographic enrichment cache for blind LLM annotation.

For each unique city/country pair in the validation set, queries EuropePMC
for proteomics publication counts and compiles research context.
Results are cached to avoid repeated API calls.
"""

import json
import time
import urllib.parse
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
LLM_CORRECTIONS = DATA_DIR / "llm_corrections.csv"
ENRICHMENT_CACHE = DATA_DIR / "location_enrichment.json"

# Known major proteomics institutions (partial list for enrichment)
KNOWN_INSTITUTIONS = {
    "Hinxton": ["EMBL-EBI", "Wellcome Sanger Institute"],
    "Heidelberg": ["EMBL Heidelberg", "German Cancer Research Center (DKFZ)"],
    "Munich": ["Max Planck Institute of Biochemistry", "Technical University of Munich"],
    "Copenhagen": ["Novo Nordisk Foundation Center for Protein Research (CPR)"],
    "Zurich": ["ETH Zurich", "University of Zurich"],
    "Seattle": ["Institute for Systems Biology", "University of Washington"],
    "Boston": ["Broad Institute", "Harvard Medical School"],
    "Cambridge": ["EMBL-EBI", "University of Cambridge", "MIT", "Broad Institute"],
    "Ghent": ["VIB-UGent Center for Medical Biotechnology"],
    "Barcelona": ["Centre for Genomic Regulation (CRG)", "Proteomics Unit UPF"],
    "Beijing": ["BGI Genomics", "Chinese Academy of Sciences"],
    "Shanghai": ["Shanghai Institute of Biochemistry"],
    "Tokyo": ["RIKEN", "University of Tokyo"],
    "Singapore": ["Nanyang Technological University", "National University of Singapore"],
    "Oxford": ["University of Oxford", "Target Discovery Institute"],
    "Lund": ["Lund University"],
    "Uppsala": ["Uppsala University", "SciLifeLab"],
    "Stockholm": ["Karolinska Institutet", "SciLifeLab"],
    "Bethesda": ["NIH/NHLBI", "National Cancer Institute"],
    "San Francisco": ["UCSF"],
    "Melbourne": ["Walter and Eliza Hall Institute"],
    "Sydney": ["Children's Medical Research Institute"],
    "Edinburgh": ["University of Edinburgh"],
    "Frankfurt": ["Goethe University Frankfurt"],
    "Incheon": ["Korea University"],
    "Daejeon": ["KAIST", "Korea Basic Science Institute"],
    "Vienna": ["CeMM", "University of Vienna"],
    "Budapest": ["Semmelweis University"],
    "Bordeaux": ["University of Bordeaux"],
    "Lausanne": ["EPFL", "University of Lausanne"],
    "Geneva": ["University of Geneva"],
    "Strasbourg": ["IPHC-CNRS"],
    "Grenoble": ["CEA Grenoble", "IBS"],
    "Aarhus": ["Aarhus University"],
    "Odense": ["University of Southern Denmark"],
    "Lyon": ["CNRS Lyon"],
    "Paris": ["Institut Pasteur", "Institut Curie"],
    "London": ["Francis Crick Institute", "Imperial College", "UCL"],
    "New York": ["Columbia University", "Memorial Sloan Kettering", "Rockefeller University"],
    "Los Angeles": ["UCLA", "Cedars-Sinai"],
    "Chicago": ["Northwestern University"],
    "Pittsburgh": ["University of Pittsburgh"],
    "Philadelphia": ["University of Pennsylvania", "Wistar Institute"],
    "Baltimore": ["Johns Hopkins University"],
    "Houston": ["Baylor College of Medicine", "MD Anderson"],
    "San Diego": ["Scripps Research", "Salk Institute"],
    "Guangzhou": ["Sun Yat-sen University"],
    "Wuhan": ["Wuhan University", "Huazhong University"],
    "Nanjing": ["Nanjing University"],
    "Taipei": ["Academia Sinica"],
    "Toronto": ["University of Toronto"],
    "Montreal": ["McGill University"],
    "Sao Paulo": ["University of Sao Paulo"],
    "Buenos Aires": ["University of Buenos Aires"],
    "Mumbai": ["IIT Bombay"],
    "Bangalore": ["Indian Institute of Science"],
}


def query_europepmc(city: str, country: str, max_retries: int = 3) -> int:
    """Query EuropePMC for proteomics papers mentioning a city."""
    query = f'(proteomics OR "mass spectrometry" OR proteome) AND "{city}"'
    encoded = urllib.parse.quote(query)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded}&resultType=lite&pageSize=1&format=json"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DeepLogBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("hitCount", 0)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  WARNING: EuropePMC query failed for {city}, {country}: {e}")
                return -1
    return -1


def build_enrichment(df: pd.DataFrame) -> dict:
    """Build enrichment data for all unique city/country pairs."""
    pairs = df[["city", "country"]].drop_duplicates().reset_index(drop=True)
    print(f"Building enrichment for {len(pairs)} unique city/country pairs...")

    # Country-level PRIDE submission counts (from location data)
    country_location_counts = df.groupby("country").size().to_dict()

    enrichment = {}
    for idx, row in pairs.iterrows():
        city = row["city"]
        country = row["country"]
        key = f"{city}|{country}"

        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(pairs)}...")

        # EuropePMC paper count
        paper_count = query_europepmc(city, country)

        # Known institutions
        institutions = KNOWN_INSTITUTIONS.get(city, [])

        # Country location count in validation set
        country_locs = country_location_counts.get(country, 0)

        # Research level assessment
        if paper_count > 100:
            research_level = "major proteomics research hub"
        elif paper_count > 20:
            research_level = "active proteomics research"
        elif paper_count > 5:
            research_level = "some proteomics research"
        elif paper_count > 0:
            research_level = "limited proteomics research"
        elif paper_count == 0:
            research_level = "no proteomics publications found"
        else:
            research_level = "unknown (query failed)"

        enrichment[key] = {
            "city": city,
            "country": country,
            "europepmc_paper_count": paper_count,
            "known_institutions": institutions,
            "country_locations_in_dataset": country_locs,
            "research_level": research_level,
        }

        # Rate limit: ~2 requests/sec to be polite to EuropePMC
        time.sleep(0.5)

    return enrichment


def format_enrichment_text(enr: dict) -> str:
    """Format enrichment data as text for LLM prompt."""
    lines = [f"Research context for {enr['city']}, {enr['country']}:"]

    if enr["europepmc_paper_count"] >= 0:
        lines.append(
            f"- Proteomics publications mentioning this city (EuropePMC): "
            f"~{enr['europepmc_paper_count']} papers"
        )
    else:
        lines.append("- Proteomics publications: unknown (query unavailable)")

    if enr["known_institutions"]:
        lines.append(
            f"- Known research institutions: {', '.join(enr['known_institutions'])}"
        )
    else:
        lines.append("- Known research institutions: none in our reference list")

    lines.append(f"- Research assessment: {enr['research_level']}")
    lines.append(
        f"- Locations from {enr['country']} in dataset: "
        f"{enr['country_locations_in_dataset']}"
    )

    return "\n".join(lines)


def main():
    df = pd.read_csv(LLM_CORRECTIONS)
    print(f"Loaded {len(df)} locations from {LLM_CORRECTIONS}")

    # Check for existing cache (resume support)
    if ENRICHMENT_CACHE.exists():
        with open(ENRICHMENT_CACHE) as f:
            existing = json.load(f)
        print(f"Found existing cache with {len(existing)} entries")

        # Check if all pairs are covered
        pairs = df[["city", "country"]].drop_duplicates()
        missing = []
        for _, row in pairs.iterrows():
            key = f"{row['city']}|{row['country']}"
            if key not in existing:
                missing.append(row)

        if not missing:
            print("All city/country pairs already cached. Nothing to do.")
            # Add formatted text to each entry
            for key, enr in existing.items():
                if "enrichment_text" not in enr:
                    enr["enrichment_text"] = format_enrichment_text(enr)
            with open(ENRICHMENT_CACHE, "w") as f:
                json.dump(existing, f, indent=2)
            return

        print(f"  {len(missing)} pairs missing, will query those only")
        missing_df = pd.DataFrame(missing)
        new_enrichment = build_enrichment(missing_df)
        existing.update(new_enrichment)
        enrichment = existing
    else:
        enrichment = build_enrichment(df)

    # Add formatted text
    for key, enr in enrichment.items():
        enr["enrichment_text"] = format_enrichment_text(enr)

    # Save
    with open(ENRICHMENT_CACHE, "w") as f:
        json.dump(enrichment, f, indent=2)

    print(f"\nSaved enrichment cache to {ENRICHMENT_CACHE}")
    print(f"  Total entries: {len(enrichment)}")

    # Stats
    paper_counts = [
        e["europepmc_paper_count"]
        for e in enrichment.values()
        if e["europepmc_paper_count"] >= 0
    ]
    if paper_counts:
        print(f"  EuropePMC query success: {len(paper_counts)}/{len(enrichment)}")
        print(f"  Paper count range: {min(paper_counts)}-{max(paper_counts)}")
        print(
            f"  Cities with >100 papers: "
            f"{sum(1 for c in paper_counts if c > 100)}"
        )
        print(
            f"  Cities with 0 papers: "
            f"{sum(1 for c in paper_counts if c == 0)}"
        )


if __name__ == "__main__":
    main()
