#!/usr/bin/env python

from greycloud import GreyCloudConfig, GreyCloudClient
from google.genai import types

# gets project, region, service account, use_api_key from environment

# Create configuration
config = GreyCloudConfig(
    model="gemini-3-flash-preview"
)

# Create client
client = GreyCloudClient(config)

# Generate content
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello, how are you?")]
    )
]

response = client.generate_content(contents)
print(response.text)

