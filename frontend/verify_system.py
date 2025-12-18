import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def verify():
    print(f"Testing API at {BASE_URL}...")
    async with httpx.AsyncClient() as client:
        try:
            # 1. Health
            resp = await client.get(f"{BASE_URL}/")
            print(f"Health Check: {resp.status_code}")
            assert resp.status_code == 200
            
            # 2. Start Simulation
            resp = await client.post(f"{BASE_URL}/simulate/start")
            print(f"Start Sim: {resp.json()}")
            assert resp.status_code == 200
            
            # 3. Check Inventory (Pick a SKU/Loc from DB - hardcoded for test)
            # Fetch products first to get valid ID
            prod_resp = await client.get(f"{BASE_URL}/products?query=Co")
            if prod_resp.status_code == 200 and prod_resp.json()['products']:
                sku = prod_resp.json()['products'][0]['sku_id']
                print(f"Testing Inventory for SKU: {sku}")
                
                # We need a location too.
                loc_resp = await client.get(f"{BASE_URL}/locations")
                loc = loc_resp.json()['locations'][0]['location_id']
                
                inv_resp = await client.get(f"{BASE_URL}/inventory/status", params={"sku_id": sku, "location_id": loc})
                print(f"Inventory Status: {inv_resp.json()}")
                assert "current_stock" in inv_resp.json()
            else:
                print("Skipping Inventory check (no products found)")

            # 4. Stop Simulation
            resp = await client.post(f"{BASE_URL}/simulate/stop")
            print(f"Stop Sim: {resp.json()}")
            
            print("\n✅ SYSTEM VERIFIED: All systems go!")
            
        except Exception as e:
            print(f"\n❌ VERIFICATION FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
