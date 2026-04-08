"""
Anthropic-to-OpenAI Proxy Server

Translates Anthropic Messages API calls to OpenAI-compatible API calls.
This allows SAGE's Claude-dependent code to use a local vLLM model instead.

Usage:
    python anthropic_proxy.py --port 8300 --backend http://localhost:8100/v1
    
Then set:
    ANTHROPIC_API_KEY=proxy-key
    ANTHROPIC_BASE_URL=http://localhost:8300
"""

import json
import sys
import os
import argparse
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

parser = argparse.ArgumentParser(description='Anthropic-to-OpenAI API Proxy')
parser.add_argument('--port', type=int, default=8300, help='Port to run proxy on')
parser.add_argument('--backend', type=str, default='http://localhost:8100/v1',
                    help='OpenAI-compatible backend URL')
parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                    help='Model name to use on backend')
parser.add_argument('--max-tokens-cap', type=int, default=4096,
                    help='Cap max_tokens to this value')
parser.add_argument('--api-key', type=str, default=None,
                    help='API key for backend authentication')
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

BACKEND_URL = args.backend
BACKEND_MODEL = args.model
MAX_TOKENS_CAP = args.max_tokens_cap
BACKEND_API_KEY = args.api_key or os.environ.get('NVIDIA_API_KEY', '')

request_count = 0
error_count = 0


def convert_anthropic_to_openai(anthropic_request: dict) -> dict:
    """Convert Anthropic Messages API request to OpenAI Chat Completions format."""
    messages = []
    
    # Handle system message
    system = anthropic_request.get('system', '')
    if system:
        if isinstance(system, list):
            # Anthropic system can be a list of content blocks
            system_text = ' '.join(
                block.get('text', '') for block in system 
                if isinstance(block, dict) and block.get('type') == 'text'
            )
        else:
            system_text = str(system)
        messages.append({"role": "system", "content": system_text})
    
    # Convert messages
    for msg in anthropic_request.get('messages', []):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle content blocks (text, image, tool_use, tool_result)
            text_parts = []
            openai_content = []
            
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get('type', '')
                    
                    if block_type == 'text':
                        text_parts.append(block.get('text', ''))
                    elif block_type == 'image':
                        # Convert Anthropic image format to OpenAI
                        source = block.get('source', {})
                        if source.get('type') == 'base64':
                            media_type = source.get('media_type', 'image/png')
                            data = source.get('data', '')
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                    elif block_type == 'tool_use':
                        # Tool calls from assistant - append as text for simplicity
                        text_parts.append(
                            f"[Tool call: {block.get('name', 'unknown')}({json.dumps(block.get('input', {}))})]"
                        )
                    elif block_type == 'tool_result':
                        # Tool results
                        tool_content = block.get('content', '')
                        if isinstance(tool_content, list):
                            tool_content = ' '.join(
                                b.get('text', '') for b in tool_content 
                                if isinstance(b, dict)
                            )
                        text_parts.append(f"[Tool result: {tool_content}]")
            
            if text_parts:
                if openai_content:
                    openai_content.insert(0, {"type": "text", "text": '\n'.join(text_parts)})
                    messages.append({"role": role, "content": openai_content})
                else:
                    messages.append({"role": role, "content": '\n'.join(text_parts)})
            elif openai_content:
                messages.append({"role": role, "content": openai_content})
        else:
            messages.append({"role": role, "content": str(content)})
    
    # Build OpenAI request
    max_tokens = min(
        anthropic_request.get('max_tokens', 1024),
        MAX_TOKENS_CAP
    )
    
    openai_request = {
        "model": BACKEND_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.get('temperature', 0.7),
    }
    
    # Handle stop sequences
    stop = anthropic_request.get('stop_sequences', [])
    if stop:
        openai_request['stop'] = stop
    
    return openai_request


def convert_openai_to_anthropic(openai_response: dict, model_name: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_response.get('choices', [{}])[0]
    message = choice.get('message', {})
    content_text = message.get('content', '')
    
    usage = openai_response.get('usage', {})
    
    return {
        "id": f"msg_{openai_response.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": content_text
            }
        ],
        "model": model_name,
        "stop_reason": "end_turn" if choice.get('finish_reason') == 'stop' else "max_tokens",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get('prompt_tokens', 0),
            "output_tokens": usage.get('completion_tokens', 0)
        }
    }


@app.route('/v1/messages', methods=['POST'])
def messages():
    """Handle Anthropic Messages API calls."""
    global request_count, error_count
    request_count += 1
    
    try:
        anthropic_request = request.get_json()
        model_requested = anthropic_request.get('model', 'claude-3-5-sonnet')
        
        print(f"[Proxy] Request #{request_count}: model={model_requested}, "
              f"max_tokens={anthropic_request.get('max_tokens', '?')}, "
              f"messages={len(anthropic_request.get('messages', []))}", 
              file=sys.stderr)
        
        # Convert to OpenAI format
        openai_request = convert_anthropic_to_openai(anthropic_request)
        
        # Forward to backend with auth
        headers = {"Content-Type": "application/json"}
        if BACKEND_API_KEY:
            headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
        
        response = requests.post(
            f"{BACKEND_URL}/chat/completions",
            json=openai_request,
            headers=headers,
            timeout=300
        )
        
        if response.status_code != 200:
            error_count += 1
            print(f"[Proxy] Backend error {response.status_code}: {response.text[:200]}",
                  file=sys.stderr)
            return jsonify({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Backend error: {response.text[:500]}"
                }
            }), 500
        
        openai_response = response.json()
        anthropic_response = convert_openai_to_anthropic(openai_response, model_requested)
        
        print(f"[Proxy] Response: {anthropic_response['usage']['output_tokens']} tokens",
              file=sys.stderr)
        
        return jsonify(anthropic_response)
    
    except Exception as e:
        error_count += 1
        print(f"[Proxy] Error: {e}", file=sys.stderr)
        return jsonify({
            "type": "error",
            "error": {
                "type": "api_error", 
                "message": str(e)
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return available models in Anthropic format."""
    return jsonify({
        "data": [
            {"id": "claude-3-5-sonnet-20241022", "type": "model"},
            {"id": "claude-sonnet-4-20250514", "type": "model"},
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "backend": BACKEND_URL,
        "model": BACKEND_MODEL,
        "requests": request_count,
        "errors": error_count
    })


if __name__ == '__main__':
    print(f"{'='*50}", file=sys.stderr)
    print(f"🔀 Anthropic-to-OpenAI Proxy — port {args.port}", file=sys.stderr)
    print(f"   Backend: {BACKEND_URL}", file=sys.stderr)
    print(f"   Model: {BACKEND_MODEL}", file=sys.stderr)
    print(f"   Max tokens cap: {MAX_TOKENS_CAP}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    app.run(host='0.0.0.0', port=args.port, debug=False)
