# Bing Search Retriever

# libraries
import os
import logging

from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import BingGroundingTool
from azure.identity import DefaultAzureCredential
from certifi import contents
from dotenv import load_dotenv


class BingSearch():
    """
    Bing Search Retriever
    """

    def __init__(self, query, query_domains=None):
        """
        Initializes the BingSearch object using Azure AI Project Client and BingGroundingTool.
        Args:
            query: The search query string.
            query_domains: Optional list of domains to restrict the search.
        """
        load_dotenv()
        self.query = query
        self.query_domains = query_domains or None
        self.logger = logging.getLogger(__name__)

        # Environment variables required for Azure AI Project Client
        self.project_endpoint = os.environ.get("AI_PROJECT_ENDPOINT")
        self.bing_connection_name = os.environ.get("BING_CONNECTION_NAME_ENV")
        if not self.project_endpoint or not self.bing_connection_name:
            raise Exception("Missing PROJECT_CONNECTION_STRING_ENV or BING_CONNECTION_NAME_ENV environment variables.")

        self.credential = DefaultAzureCredential()
        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=DefaultAzureCredential(),  # Use Azure Default Credential for authentication
            api_version="2025-05-15-preview"
        )  
        
        self.bing_tool = BingGroundingTool(connection_id=self.bing_connection_name)
        
    def search(self, max_results=7) -> dict:
        # Build query string, supporting query_domains as site: filters
        query = self.query
        if self.query_domains:
            if isinstance(self.query_domains, list):
                domains = [f"site:{d}" for d in self.query_domains]
                query += " " + " OR ".join(domains)
            else:
                query += f" site:{self.query_domains}"

        agent = None
        try:
            with self.project_client as client:

                # Step 1: Create agent with BingGroundingTool if not exists
                #agent = client.agents.get_agent("asst_Y9foDfJksj66HXjyJpEbShuR")
                agent = client.agents.create_agent(
                    model="gpt-4.1-mini",
                    name="gpt-researcher-bing-agent",
                    instructions="You are a helpful agent for Bing search.",
                    tools=self.bing_tool.definitions,
                    headers={"x-ms-enable-preview": "true"}
                )

                # Step 2: Create conversation thread
                thread = client.agents.threads.create()
                self.logger.info(f"Created thread with id: {thread.id}")
                # Step 3: Add user message to the thread
                user_message = client.agents.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=query
                )

                # Step 4: Run the agent
                run = client.agents.runs.create_and_process(
                    thread_id=thread.id,
                    agent_id=agent.id
                )

                if run.status == "failed":
                    self.logger.error(f"Run failed: {run.last_error}")
                    return []
                else:
                    # Step 5: Retrieve agent's response
                    messages = list(client.agents.messages.list(thread_id=thread.id))

                    last_msg = None
                    for msg in reversed(messages):
                        if getattr(msg, "role", None) == "assistant":
                            last_msg = msg
                            break

                    # Build response in the new format
                    response_list = []
                    if last_msg and hasattr(last_msg, "text") and hasattr(last_msg.text, "value"):
                        value = last_msg.text.value
                        # Defensive: get annotations if present
                        annotations = getattr(last_msg.text, "annotations", [])
                        formatted_annotations = []
                        for annotation in annotations:
                            # Defensive: get all required fields, fallback to None if missing
                            ann_type = getattr(annotation, "type", None)
                            ann_text = getattr(annotation, "text", None)
                            start_index = getattr(annotation, "start_index", None)
                            end_index = getattr(annotation, "end_index", None)
                            url_citation = getattr(annotation, "url_citation", None)
                            url = getattr(url_citation, "url", None) if url_citation else None
                            title = getattr(url_citation, "title", None) if url_citation else None
                            formatted_annotations.append({
                                "type": ann_type,
                                "text": ann_text,
                                "start_index": start_index,
                                "end_index": end_index,
                                "url_citation": {
                                    "url": url,
                                    "title": title
                                } if url_citation else None
                            })
                        response_list.append({
                            "type": "text",
                            "text": {
                                "value": value,
                                "annotations": formatted_annotations
                            }
                        })
                    else:
                        self.logger.warning("No response from the agent.")

                # Always attempt to delete the agent before returning
                try:
                    if agent is not None:
                        client.agents.delete_agent(agent.id)
                        self.logger.info(f"Deleted agent with id: {agent.id}")
                except Exception as del_exc:
                    self.logger.warning(f"Failed to delete agent: {del_exc}")

                return response_list
        except Exception as e:
            self.logger.error(f"Azure Bing Search failed: {e}")
            return []
