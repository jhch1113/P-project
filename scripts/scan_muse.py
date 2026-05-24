import asyncio
from bleak import BleakScanner

async def run():
    print("Scanning for Bluetooth devices (Please turn on your Muse device)...")
    devices = await BleakScanner.discover(timeout=5.0)
    found = False
    for d in devices:
        if d.name and "Muse" in d.name:
            print(f"\n[Success] Found Device Name : {d.name}")
            print(f"[Success] Device MAC Address : {d.address}\n")
            found = True
    if not found:
        print("\nCould not find the Muse device. Please check the power and pairing status.\n")

if __name__ == "__main__":
    asyncio.run(run())
