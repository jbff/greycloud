#!/usr/bin/env python3
"""
Example: Using GreyCloud Context Caching

This example demonstrates how to use context caching to reduce costs when
making multiple queries against the same large content.

Context caching provides:
- 75-90% discount on cached input tokens (depending on model)
- Storage cost of $1.00 per million tokens per hour

Requirements:
- Set PROJECT_ID environment variable
- Authenticate with: gcloud auth application-default login
- The content must meet minimum token threshold (1,024-4,096 tokens)

Usage:
    python examples/caching.py
"""

import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greycloud import GreyCloudConfig, GreyCloudCache
from google.genai import types


def main():
    """Demonstrate context caching usage"""

    # Create configuration
    config = GreyCloudConfig(
        # project_id is read from PROJECT_ID env var by default
        # model defaults to gemini-3-flash-preview
    )

    print(f"Project: {config.project_id}")
    print(f"Model: {config.model}")
    print()

    # Create cache client
    cache_client = GreyCloudCache(config)

    # Create some large content to cache
    # Note: Content must meet minimum token threshold (1,024+ tokens for Flash models)
    large_document = """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can access data and use it to learn
    for themselves.

    ## Types of Machine Learning

    ### 1. Supervised Learning
    In supervised learning, the algorithm learns from labeled training data,
    and makes predictions based on that data. Examples include:
    - Classification: Categorizing data into predefined classes
    - Regression: Predicting continuous values

    ### 2. Unsupervised Learning
    Unsupervised learning works with unlabeled data. The algorithm tries to
    find patterns and relationships in the data. Examples include:
    - Clustering: Grouping similar data points
    - Dimensionality reduction: Reducing the number of features

    ### 3. Reinforcement Learning
    Reinforcement learning involves an agent learning to make decisions by
    taking actions in an environment to maximize cumulative reward.

    ## Key Concepts

    ### Features and Labels
    Features are the input variables used to make predictions. Labels are
    the output variables we want to predict (in supervised learning).

    ### Training and Testing
    Data is typically split into training and testing sets. The model learns
    from training data and is evaluated on testing data to measure performance.

    ### Overfitting and Underfitting
    - Overfitting: The model learns the training data too well, including noise
    - Underfitting: The model is too simple to capture the underlying patterns

    ### Hyperparameters
    Hyperparameters are configuration settings used to tune how the model learns.
    They are set before training begins, unlike model parameters which are learned.

    ## Common Algorithms

    1. Linear Regression
    2. Logistic Regression
    3. Decision Trees
    4. Random Forests
    5. Support Vector Machines
    6. Neural Networks
    7. K-Nearest Neighbors
    8. K-Means Clustering
    9. Principal Component Analysis

    ## Applications

    Machine learning is used in many real-world applications:
    - Image and speech recognition
    - Natural language processing
    - Recommendation systems
    - Fraud detection
    - Medical diagnosis
    - Autonomous vehicles
    - Stock market prediction
    - Customer segmentation

    ## Best Practices

    1. Start with simple models before complex ones
    2. Use cross-validation to evaluate model performance
    3. Handle missing data appropriately
    4. Normalize or standardize features when necessary
    5. Monitor for data drift in production
    6. Document your experiments and results
    7. Consider model interpretability requirements
    8. Regularly retrain models with new data
    """ * 5  # Repeat to ensure we meet minimum token threshold

    print("=" * 60)
    print("STEP 1: Create a cache")
    print("=" * 60)

    try:
        # Create cache with the document
        cache = cache_client.create_cache_from_text(
            text=large_document,
            display_name="ml-tutorial-cache",
            system_instruction="You are a helpful AI tutor specializing in machine learning.",
            ttl_seconds=300,  # 5 minutes for demo purposes
        )

        print(f"Cache created: {cache.name}")
        if hasattr(cache, 'expire_time'):
            print(f"Expires at: {cache.expire_time}")
        if hasattr(cache, 'usage_metadata') and cache.usage_metadata:
            print(f"Cached tokens: {cache.usage_metadata.total_token_count}")
        print()

    except Exception as e:
        print(f"Error creating cache: {e}")
        print("\nNote: Caching requires:")
        print("  - Valid Google Cloud credentials")
        print("  - Content that meets minimum token threshold")
        print("  - Paid API access (not available in free tier)")
        return

    print("=" * 60)
    print("STEP 2: Query the cache multiple times")
    print("=" * 60)

    # Ask multiple questions using the same cached content
    questions = [
        "What are the three main types of machine learning?",
        "Explain overfitting in simple terms.",
        "List 5 real-world applications of machine learning.",
        "What is the difference between features and labels?",
    ]

    try:
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 40)

            response = cache_client.generate_with_cache(
                cache_name=cache.name,
                prompt=question,
                temperature=0.7,
                max_output_tokens=500,
            )

            print(f"Answer: {response.text[:500]}...")
            if len(response.text) > 500:
                print("[Response truncated]")
            print()

    except Exception as e:
        print(f"Error generating content: {e}")

    print("=" * 60)
    print("STEP 3: List all caches")
    print("=" * 60)

    try:
        print("\nCurrent caches:")
        for cached_content in cache_client.list_caches():
            info = cache_client.get_cache_info(cached_content)
            print(f"  - {info['name']}")
            if info.get('display_name'):
                print(f"    Display name: {info['display_name']}")
            if info.get('total_token_count'):
                print(f"    Tokens: {info['total_token_count']}")
        print()

    except Exception as e:
        print(f"Error listing caches: {e}")

    print("=" * 60)
    print("STEP 4: Delete the cache (important to avoid storage costs!)")
    print("=" * 60)

    try:
        cache_client.delete_cache(cache.name)
        print(f"Cache deleted: {cache.name}")
        print("\nStorage charges have stopped for this cache.")

    except Exception as e:
        print(f"Error deleting cache: {e}")

    print()
    print("=" * 60)
    print("COST SAVINGS SUMMARY")
    print("=" * 60)
    print("""
When using context caching effectively:

1. CACHED TOKEN DISCOUNT:
   - Gemini 2.5+ models: 90% discount (pay only 10% of standard rate)
   - Gemini 2.0 models: 75% discount (pay only 25% of standard rate)

2. STORAGE COST:
   - $1.00 per million tokens per hour (prorated by minute)

3. BEST USE CASES:
   - Multiple queries on the same large document
   - Chatbots with extensive system instructions
   - Repeated analysis of the same media files
   - Code repository analysis with recurring queries

4. TIPS:
   - Delete caches when done to stop storage charges
   - Use appropriate TTL values for your use case
   - Cache content that will be queried multiple times
   - Consider the break-even point: storage cost vs token savings
""")


def demo_streaming():
    """Demonstrate streaming with cached content"""

    config = GreyCloudConfig()
    cache_client = GreyCloudCache(config)

    # Create a simple cache
    content = "Python is a high-level programming language known for its simplicity." * 100

    try:
        cache = cache_client.create_cache_from_text(
            text=content,
            display_name="streaming-demo",
            ttl_seconds=60,
        )

        print("Streaming response:")
        print("-" * 40)

        for chunk in cache_client.generate_with_cache_stream(
            cache_name=cache.name,
            prompt="Describe Python's key features in detail.",
        ):
            print(chunk, end="", flush=True)

        print("\n")

        # Clean up
        cache_client.delete_cache(cache.name)
        print("Cache deleted.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("GreyCloud Context Caching Example")
    print("=" * 60)
    print()

    main()

    # Uncomment to run streaming demo:
    # demo_streaming()
