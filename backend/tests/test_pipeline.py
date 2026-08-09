"""
VoxPilot RAG Pipeline Test Script

Usage:
    python tests/test_pipeline.py [--base-url http://localhost:8000]
    
Requires:
    - OPENAI_API_KEY env variable
    - A running VoxPilot backend + services
"""

import argparse
import os
import sys
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


async def test_auth(client: httpx.AsyncClient, email: str, password: str, name: str):
    print_section("1. Auth: Register & Login")

    resp = await client.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    if resp.status_code == 409:
        print("  User already exists, logging in...")
        resp = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json={"email": email, "password": password},
        )
    elif resp.status_code != 201:
        print(f"  Register failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    else:
        print(f"  Registered: {name} <{email}>")

    data = resp.json()
    token = data.get("access_token")
    if not token:
        print(f"  Login failed: {data}")
        sys.exit(1)

    client.headers["Authorization"] = f"Bearer {token}"
    print(f"  Login successful. Token obtained.")

    me_resp = await client.get(f"{BASE_URL}/api/v1/auth/me")
    if me_resp.status_code == 200:
        user = me_resp.json()
        print(f"  Authenticated as {user['name']} (tenant_id={user['tenant_id']})")

    return token


async def test_organization(client: httpx.AsyncClient):
    print_section("2. Organization")

    resp = await client.get(f"{BASE_URL}/api/v1/organizations/me")
    if resp.status_code == 200:
        org = resp.json()
        print(f"  Organization: {org['name']} (id={org['id']})")

    resp = await client.patch(
        f"{BASE_URL}/api/v1/organizations/me",
        json={
            "name": "Acme Corp",
            "description": "Provider of cutting-edge solutions",
            "industry": "Technology",
        },
    )
    if resp.status_code == 200:
        org = resp.json()
        print(f"  Updated: {org['name']} - {org['description']}")


async def test_document_upload(client: httpx.AsyncClient, pdf_path: str | None = None):
    print_section("3. Document Upload")

    if pdf_path and os.path.exists(pdf_path):
        filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            resp = await client.post(
                f"{BASE_URL}/api/v1/documents",
                files={"file": (filename, f, "application/pdf")},
            )
        if resp.status_code == 201:
            doc = resp.json()
            print(f"  Uploaded: {doc['original_filename']} (status={doc['status']})")
            return doc["id"]
        else:
            print(f"  Upload failed: {resp.status_code} {resp.text}")
            return None
    else:
        print("  No test PDF provided. Skipping upload test.")
        if pdf_path:
            print(f"  (File not found: {pdf_path})")
        return None


async def test_document_list(client: httpx.AsyncClient):
    print_section("4. Document List")

    resp = await client.get(f"{BASE_URL}/api/v1/documents")
    if resp.status_code == 200:
        docs = resp.json()
        print(f"  Documents: {len(docs)}")
        for doc in docs:
            print(f"    - {doc['original_filename']} [{doc['status']}] chunks={doc['chunk_count']}")


async def test_document_status(client: httpx.AsyncClient, doc_id: int, timeout: int = 60):
    import asyncio

    print_section(f"5. Document Processing Status (doc_id={doc_id})")

    for _ in range(timeout // 2):
        resp = await client.get(f"{BASE_URL}/api/v1/documents/{doc_id}")
        if resp.status_code == 200:
            doc = resp.json()
            status = doc["status"]
            chunks = doc.get("chunk_count", 0)
            error = doc.get("error_message", "")
            print(f"  Status: {status}, chunks: {chunks}")

            if status == "completed":
                print(f"  Processing complete.")
                return True
            elif status == "failed":
                print(f"  Processing failed: {error}")
                return False

            await asyncio.sleep(2)
        else:
            print(f"  Failed to check status: {resp.status_code}")
            return False

    print("  Timed out waiting for processing.")
    return False


async def test_rag_search(client: httpx.AsyncClient, query: str = "What is the refund policy?"):
    print_section(f"6. RAG Search: \"{query}\"")

    resp = await client.post(
        f"{BASE_URL}/api/v1/knowledge/search",
        json={"query": query},
    )
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", [])
        print(f"  Results: {len(results)}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['document_name']} (page {r['page_number']}, score={r['score']:.4f})")
            print(f"      {r['content'][:120]}...")
        return results
    else:
        print(f"  Search failed: {resp.status_code} {resp.text}")
        return []


async def test_rag_ask(client: httpx.AsyncClient, query: str = "What is the refund policy?"):
    print_section(f"7. RAG Ask: \"{query}\"")

    resp = await client.post(
        f"{BASE_URL}/api/v1/knowledge/ask",
        json={"query": query},
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Answer: {data['answer']}")
        print(f"  Sources: {len(data.get('sources', []))}")
        for s in data.get("sources", []):
            print(f"    - {s['document_name']} (page {s['page_number']}, score={s['score']:.4f})")
        return data
    else:
        print(f"  Ask failed: {resp.status_code} {resp.text}")
        return {}


async def test_chat(client: httpx.AsyncClient, query: str = "What services do you offer?"):
    print_section(f"8. Chat: \"{query}\"")

    resp = await client.post(
        f"{BASE_URL}/api/v1/chat",
        json={"message": query},
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Conversation ID: {data.get('conversation_id')}")
        print(f"  Answer: {data['answer']}")
        return data.get("conversation_id")
    else:
        print(f"  Chat failed: {resp.status_code} {resp.text}")
        return None


async def test_conversation_followup(client: httpx.AsyncClient, conv_id: int, query: str):
    print_section(f"9. Chat Follow-up (conv={conv_id}): \"{query}\"")

    resp = await client.post(
        f"{BASE_URL}/api/v1/chat",
        json={"conversation_id": conv_id, "message": query},
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Answer: {data['answer']}")
        return data
    else:
        print(f"  Chat failed: {resp.status_code} {resp.text}")
        return None


async def test_conversation_list(client: httpx.AsyncClient):
    print_section("10. Conversation List")

    resp = await client.get(f"{BASE_URL}/api/v1/chat/conversations")
    if resp.status_code == 200:
        convs = resp.json()
        print(f"  Conversations: {len(convs)}")
        for c in convs:
            msg_count = len(c.get("messages", []))
            print(f"    - {c.get('title', 'Untitled')[:50]} ({msg_count} messages)")
    else:
        print(f"  Failed: {resp.status_code}")


async def main():
    parser = argparse.ArgumentParser(description="VoxPilot RAG Pipeline Test")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--pdf", help="Path to a test PDF file")
    parser.add_argument("--email", default="admin@test.com")
    parser.add_argument("--password", default="testpass123")
    parser.add_argument("--name", default="Test Admin")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        await test_auth(client, args.email, args.password, args.name)
        await test_organization(client)
        doc_id = await test_document_upload(client, args.pdf)
        await test_document_list(client)

        if doc_id:
            success = await test_document_status(client, doc_id)
            if success:
                await test_rag_search(client)
                await test_rag_ask(client)
                conv_id = await test_chat(client, "What is the refund policy?")
                if conv_id:
                    await test_conversation_followup(client, conv_id, "What about international orders?")
                await test_conversation_list(client)
            else:
                print("\n  Document processing failed. Skipping RAG tests.")
        else:
            print("\n  No document uploaded. Skipping RAG tests (upload a document first).")

    print_section("All tests completed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
