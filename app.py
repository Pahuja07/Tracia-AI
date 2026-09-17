import argparse

from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.agent import CriminalNetworkAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the criminal-network analysis agent.")
    parser.add_argument("query", nargs="?", help="The investigator's question.")
    parser.add_argument(
        "--entity-focus",
        help="Optional entity whose graph connections should be included in the answer.",
    )
    args = parser.parse_args()

    # Prompt on every run so an old hard-coded question cannot be reused accidentally.
    user_query = args.query or input("Investigator question: ").strip()
    if not user_query:
        parser.error("A question is required.")

    entity_focus = args.entity_focus
    if entity_focus is None:
        entity_focus = input("Entity focus (optional): ").strip() or None

    config = ConfigurationManager().get_agent_config()
    agent = CriminalNetworkAgent(config=config)
    try:
        answer = agent.answer_query(
            user_query=user_query,
            entity_focus=entity_focus,
        )
        print(f"\nAnswer:\n{answer}", flush=True)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
