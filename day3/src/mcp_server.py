from pathlib import Path
from pathlib import Path
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider


mcp = FastMCP("RetajSWE Tools")

mcp.add_provider(
    SkillsDirectoryProvider(
        roots=Path(__file__).parent.parent / "skills"
    )
)
@mcp.tool
def calculate(expression: str) -> float:
    """Calculate a basic arithmetic expression."""
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression}") from exc


@mcp.tool
def word_stats(text: str) -> dict:
    """Return basic statistics about text."""
    words = text.split()

    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(text.splitlines()),
    }


skills_path = Path(__file__).resolve().parent.parent / "skills"

mcp.add_provider(
    SkillsDirectoryProvider(
        roots=[str(skills_path)]
    )
)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8001,
    )