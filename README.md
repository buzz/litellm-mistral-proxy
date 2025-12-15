# litellm-mistral-proxy

A LiteLLM proxy server with built-in message sanitization for Mistral models.

This is very handy when self-hosting Devstral 2 or other Mistral models using **llama.cpp**.

## Features

- **Mistral Message Sanitization**: Automatically enforces Mistral's conversation template constraints:
  - Maximum one system message at the start
  - Strictly alternating user/assistant roles (squashes consecutive messages)
- **Proxy Server**: Built on LiteLLM's proxy functionality
- **API Key Authentication**: Secure your proxy with API keys
- **Environment Configuration**: Easy setup via environment variables

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)

## Installation

1. Check out the repository
2. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Configuration

Create a `.env` file based on `.env.example`.

## Usage

### Running the Proxy

```bash
# Start the proxy server
./start.sh
```
