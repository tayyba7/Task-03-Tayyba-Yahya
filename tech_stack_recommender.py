```python
"""
Project 3 - AI Recommendation Logic
Capstone: Tech Stack Recommender

Method: Content-Based Filtering using TF-IDF and Cosine Similarity

Author: [Tayyba Yahya]
DecodeLabs Industrial Training Kit, Batch 2026
"""

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "raw_skills.csv"
MIN_INPUTS = 3
TOP_N = 3

# Different names for the same skill
ALIASES = {
    "web design": "ui design",
    "web designing": "ui design",
    "frontend": "frontend development",
    "ml": "machine learning",
    "ai": "machine learning",
    "artificial intelligence": "machine learning",
    "js": "javascript",
    "py": "python",
    "k8s": "kubernetes",
    "amazon web services": "aws",
    "cloud": "cloud computing",
    "cyber security": "security",
    "db": "databases",
    "dbms": "databases",
    "structured query language": "sql"
}


def normalise(skill):
    """Clean the skill name and make it easier to compare."""
    s = skill.strip().lower()
    s = ALIASES.get(s, s)
    return s.replace(" ", "_")


def to_document(skill_list):
    """Convert skills into one text document."""
    return " ".join(normalise(s) for s in skill_list if s.strip())


def load_items(path=DATA_FILE):
    """Load career data from the CSV file."""
    df = pd.read_csv(path)
    df["document"] = df["skills"].apply(
        lambda s: to_document(s.split(","))
    )
    return df


def get_user_skills():
    """Get at least 3 skills from the user."""
    print("=" * 60)
    print("TECH STACK RECOMMENDER")
    print("=" * 60)
    print(f"Enter your skills or interests (minimum {MIN_INPUTS}).")
    print("Enter one skill per line.")
    print("Press Enter on an empty line when you are finished.\n")

    skills = []

    while True:
        entry = input(f"Skill {len(skills) + 1}: ").strip()

        if entry == "":
            if len(skills) >= MIN_INPUTS:
                break
            print(f"You need at least {MIN_INPUTS} skills. Keep going.")
            continue

        skills.append(entry)

    return skills


def recommend(user_skills, items, top_n=TOP_N, show_vectors=False):
    """Find the career paths that best match the user's skills."""
    user_doc = to_document(user_skills)

    # Create TF-IDF vectors for all career paths
    vectorizer = TfidfVectorizer()
    item_matrix = vectorizer.fit_transform(items["document"])
    user_vector = vectorizer.transform([user_doc])

    # Check which user skills are available in the dataset
    known = set(vectorizer.get_feature_names_out())

    matched = [
        s for s in user_skills
        if normalise(s) in known
    ]

    unmatched = [
        s for s in user_skills
        if normalise(s) not in known
    ]

    # If none of the skills are known, use popularity instead
    if user_vector.nnz == 0:
        print("\n[COLD START] None of your skills are in the dataset.")
        print("Showing the most popular career paths instead.\n")

        fallback = items.sort_values(
            "popularity",
            ascending=False
        ).head(top_n)

        return fallback.assign(score=0.0), matched, unmatched, True

    # Compare the user profile with each career path
    scores = cosine_similarity(
        user_vector,
        item_matrix
    ).flatten()

    if show_vectors:
        print("\nTF-IDF weights for your recognised skills:")

        idf = dict(
            zip(
                vectorizer.get_feature_names_out(),
                vectorizer.idf_
            )
        )

        for skill in matched:
            token = normalise(skill)
            print(
                f"  {skill:<22} "
                f"idf={idf[token]:.3f}"
            )

    # Sort career paths from highest match to lowest
    ranked = items.assign(score=scores).sort_values(
        "score",
        ascending=False
    )

    # Remove career paths with no matching skills
    ranked = ranked[ranked["score"] > 0]

    return ranked.head(top_n), matched, unmatched, False


def display(results, matched, unmatched, cold_start, user_skills):
    """Display the user's profile and recommendations."""
    print("\n" + "=" * 60)
    print("YOUR PROFILE")
    print("=" * 60)

    print(f"Skills entered: {', '.join(user_skills)}")
    print(
        f"Recognised: "
        f"{', '.join(matched) if matched else 'none'}"
    )

    if unmatched:
        print(f"Not in dataset: {', '.join(unmatched)}")

    print("\n" + "=" * 60)
    print(f"TOP {len(results)} RECOMMENDED CAREER PATHS")
    print("=" * 60)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):

        if cold_start:
            print(f"\n{rank}. {row['role']} (popularity fallback)")
        else:
            percentage = row["score"] * 100
            bar = "#" * int(percentage / 4)

            print(
                f"\n{rank}. {row['role']} "
                f"match: {percentage:5.1f}% {bar}"
            )

        print(f"   Skills: {row['skills']}")

        if not cold_start:
            role_skills = {
                normalise(skill)
                for skill in row["skills"].split(",")
            }

            overlap = [
                skill for skill in matched
                if normalise(skill) in role_skills
            ]

            if overlap:
                print(
                    f"   You already have: "
                    f"{', '.join(overlap)}"
                )

    print()


def main():
    try:
        items = load_items()
    except FileNotFoundError:
        print(
            f"Error: {DATA_FILE} not found. "
            "Keep it in the same folder."
        )
        sys.exit(1)

    # Run demo profiles when --demo is used
    if "--demo" in sys.argv:
        profiles = [
            ["Python", "Cloud Computing", "Automation"],
            ["HTML", "CSS", "JavaScript"],
            ["knitting", "baking", "gardening"]
        ]

        for profile in profiles:
            print("\n" + "#" * 60)
            print(f"# DEMO PROFILE: {profile}")
            print("#" * 60)

            results, matched, unmatched, cold = recommend(
                profile,
                items,
                show_vectors=True
            )

            display(
                results,
                matched,
                unmatched,
                cold,
                profile
            )

        return

    user_skills = get_user_skills()

    results, matched, unmatched, cold = recommend(
        user_skills,
        items
    )

    display(
        results,
        matched,
        unmatched,
        cold,
        user_skills
    )


if __name__ == "__main__":
    main()
```
