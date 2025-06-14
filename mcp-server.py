# minimal_mcp_server.py
import asyncio
import json
import sys

async def add(a: int, b: int) -> int:
    return a + b

async def process_request(req_json):
    try:
        req = json.loads(req_json)
        method_name = req.get("method")
        params = req.get("params")
        req_id = req.get("id")

        if method_name == "add":
            result = await add(**params)
            response = {"jsonrpc": "2.0", "result": result, "id": req_id}
        else:
            response = {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": req_id}

        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush() # Crucial
    except Exception as e:
        response = {"jsonrpc": "2.0", "error": {"code": -32600, "message": str(e)}, "id": None} # Or try to get req_id
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

async def main():
    # This is a simplified stdio loop; mcp.server.stdio.stdio_transport is more robust
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break
        line_str = line_bytes.decode('utf-8').strip()
        if line_str:
            await process_request(line_str)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
