import sys
import httpx


def discover(peer_base_url):
    url = f"{peer_base_url.rstrip('/')}/.well-known/agent-card.json"

    response = httpx.get(url)
    response.raise_for_status()

    card = response.json()

    print("Agent:", card["name"])
    print("Skills:")
    for skill in card.get("skills", []):
        print(f"- {skill['name']}: {skill['description']}")

    return card


def delegate(card, task):
    response = httpx.post(
    card["url"],
    json={"input": task},
    timeout=120.0,
)
    response.raise_for_status()

    data = response.json()

    return data["output"][0]["content"][0]["text"]


if __name__ == "__main__":
    peer_url = sys.argv[1]
    task = sys.argv[2]

    card = discover(peer_url)
    result = delegate(card, task)

    print("\nAgent response:")
    print(result)