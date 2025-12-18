import httpx
import asyncio

BASE_URL = "http://localhost:8001"

async def debug_api():
    print(f"Debugging API at {BASE_URL}...")
    async with httpx.AsyncClient() as client:
        try:
            # 1. Locations
            print("Fetching Locations...")
            resp = await client.get(f"{BASE_URL}/locations")
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ Locations: {len(data.get('locations', []))} found.")
            else:
                print(f"❌ Locations Failed: {resp.status_code} - {resp.text}")

            # 2. Products
            print("Fetching Products (query='')")
            resp = await client.get(f"{BASE_URL}/products?query=")
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ Products: {len(data.get('products', []))} found.")
            else:
                print(f"❌ Products Failed: {resp.status_code} - {resp.text}")
                
        except Exception as e:
            print(f"🔥 Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_api())
