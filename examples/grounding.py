#!/usr/bin/env python

"""Vertex AI Search grounding: explicit retrieval (search-and-inject).

GreyCloud's default `grounding_mode="inject"` runs the Discovery Engine
search itself (with the same credentials used for generation) and injects
the top results into the prompt as an attributed <grounding_sources> block.
This works with every model version — the server-side tools.retrieval
grounding silently returns zero chunks on Gemini 3.x models.

Run this with real GCP credentials (ADC, SA impersonation, or an API key)
and a real datastore. A search failure degrades to generation-without-
context (logged, never an error).
"""

from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

# gets project, region, service account, use_api_key from environment

# Inject mode (the default): no retrieval tool is sent; GreyCloud searches
# the datastore and prefixes the last user message with the top sources.
config = GreyCloudConfig(
    model="gemini-3.7-flash",
    use_vertex_ai_search=True,
    vertex_ai_search_datastore=(
        "projects/PROJECT_ID/locations/us/"
        "collections/default_collection/dataStores/DATASTORE_ID"
    ),
    # grounding_mode defaults to "inject" — no need to set it
)

client = GreyCloudClient(config)

contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Using the knowledge base, quote one short sentence about autistic inertia."
            )
        ],
    )
]

response = client.generate_content(contents)
print(response.text)

# Legacy mode: keep the server-side tools.retrieval tool (no search is run,
# no grounding block is injected). Useful on model versions where the
# retrieval tool still works, e.g. gemini-2.5.
tool_mode_config = GreyCloudConfig(
    model="gemini-2.5-flash",
    use_vertex_ai_search=True,
    vertex_ai_search_datastore=(
        "projects/PROJECT_ID/locations/us/"
        "collections/default_collection/dataStores/DATASTORE_ID"
    ),
    grounding_mode="tool",
)

tool_client = GreyCloudClient(tool_mode_config)
response = tool_client.generate_content(contents)
print(response.text)
