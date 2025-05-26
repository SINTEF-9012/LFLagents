import os
from utils.sindit_api import query_get_api, query_post_api, load_json

if __name__ == "__main__":
    # 1. Register connection
    connection = load_json(os.path.join(os.path.dirname(__file__), 'config/connections.json'))
    query_post_api('kg/connection', connection)

    # 2. Register streaming properties
    streaming_properties = load_json(os.path.join(os.path.dirname(__file__), 'config/streaming_properties.json'))
    for prop in streaming_properties:
        query_post_api('kg/node', prop)
        print(f"Registered streaming property: {prop['uri']}")

    # 3. Register assets (factory sensor, camera, vgr, hbw, sld, mpo, dso, dsi, order)
    assets = load_json(os.path.join(os.path.dirname(__file__), 'config/assets.json'))
    for asset in assets:
        query_post_api('kg/asset', asset)
        print(f"Registered asset: {asset['uri']}")

    # 4. Register sindit_kg
    sindit_kg = load_json(os.path.join(os.path.dirname(__file__), 'config/sindit_kg.json'))
    query_post_api('kg/sindit_kg', sindit_kg)
