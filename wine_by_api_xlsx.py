# description: Script to fetch wine data from an API, download images, and save details to an Excel file with hyperlinks.
# requirements: pandas, openpyxl, requests
# usage: use web browser to open the website and inspect network requests to get headers and cookies. paste them cookies in the script, ie. 'cookie': 'uuvid=xxxxxxxx'
#   
import requests
import os
import time
import sys
import re
from pathlib import Path
import pandas as pd

# Set current working directory to the directory containing this script
script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path(sys.argv[0]).resolve().parent
try:
    os.chdir(script_dir)
    print(f"Working directory set to script folder: {os.path.abspath(os.getcwd())}")
except Exception as e:
    print(f"Could not change working directory: {e}")

# Display current working folder full path
cwd = os.path.abspath(os.getcwd())
print(f"Current folder: {cwd}")

# Create images directory if it doesn't exist
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)
print(f"Images directory: {images_dir.resolve()}")

# Headers and cookies for the request
headers = {
    'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
}

cookies = {
    'cookie': 'uuvid=A1BXDgA/AGsFYVVuXjELPQE0AW4JZAhoAz4CMQZuX20JNQE+C2EFMAc1ADkINQEzBjFXNQ1lWmIJP1AwVG1WZAMzV2Y=; isLogin=true; Hm_lvt_7d46a3151782b7a795ffeba367b5387d=1764905836; HMACCOUNT=F60918C6E83B73A2; Hm_lpvt_7d46a3151782b7a795ffeba367b5387d=1764905927'
}

url = "https://allinone.pospal.cn/wxapi/product/ListMulti"

# Excel columns (no change needed)
excel_columns = [
    "id", "name", "category_name", "sellPrice", "productOriginalPrice", "discount_rate", "stock",
    "description", "attribute6", "createTime", "image_name", "image_url"
]

# Process all pages
all_wine_data = []

for page in range(100):  # Pages 0 to 99
    print(f"Processing page {page + 1}...")
    
    data = {
        "cUids": 12,
        "pageIdx": page,
        "size": 50
    }
    
    try:
        response = requests.post(
            url, 
            data=data, 
            headers=headers, 
            cookies=cookies
        )
        response.raise_for_status()
        json_response = response.json()
        
        if 'data' not in json_response or not json_response['data']:
            print(f"No more data on page {page + 1}, stopping.")
            break
            
        for item in json_response['data']:
            wine_info = {
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "category_name": item.get("category", {}).get("name", ""),
                "sellPrice": item.get("sellPrice", ""),
                "stock": item.get("stock", ""),
                "productOriginalPrice": item.get("productOriginalPrice", ""),
                "description": item.get("description", ""),
                "attribute6": item.get("attribute6", ""),
                "createTime": item.get("createTime", "")
            }

            # Calculate discount_rate as sellPrice / productOriginalPrice (numeric fraction)
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            sp = _to_float(wine_info.get("sellPrice"))
            pop = _to_float(wine_info.get("productOriginalPrice"))
            discount_rate = None
            if sp is not None and pop and pop != 0:
                discount_rate = sp / pop

            # attach discount_rate to wine_info so it becomes part of output
            wine_info["discount_rate"] = discount_rate
            
            image_name = ""
            image_url_display = ""  # Cleaned URL for display
            local_image_path = ""
            web_image_url = ""      # Full web URL for hyperlink
            
            if "defaultproductimage" in item and item["defaultproductimage"]:
                original_image_path = item["defaultproductimage"].get("imagepath", "").strip()
                if original_image_path:
                    # Clean the URL: remove "_200x200" before .jpg/.jpeg
                    image_url_display = re.sub(r'(?i)_200x200(?=\.(?:jpe?g))', '', original_image_path)
                    web_image_url = image_url_display
                    
                    # Local filename
                    image_name = f"{item.get('id', '')}.jpg"
                    image_path = images_dir / image_name
                    
                    # Download image
                    try:
                        img_response = requests.get(web_image_url)
                        if img_response.status_code == 200:
                            with open(image_path, 'wb') as img_file:
                                img_file.write(img_response.content)
                            print(f"Downloaded image: {image_name}")
                            local_image_path = str(image_path.resolve())
                        else:
                            print(f"Failed to download image for ID {item.get('id')}: HTTP {img_response.status_code}")
                    except Exception as e:
                        print(f"Error downloading image for ID {item.get('id')}: {e}")
            
            # Store all data including paths for hyperlinking
            all_wine_data.append({
                "id": wine_info["id"],
                "name": wine_info["name"],
                "category_name": wine_info["category_name"],
                "sellPrice": wine_info["sellPrice"],
                "productOriginalPrice": wine_info["productOriginalPrice"],
                "discount_rate": wine_info["discount_rate"],
                "stock": wine_info["stock"],
                "description": wine_info["description"],
                "attribute6": wine_info["attribute6"],
                "createTime": wine_info["createTime"],
                "image_name": image_name,
                "image_url": image_url_display,
                "_local_image_path": local_image_path,   # Temporary: for local hyperlink
                "_web_image_url": web_image_url          # Temporary: for web hyperlink
            })
            
        time.sleep(0.5)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page + 1}: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error on page {page + 1}: {e}")
        continue

# Save to Excel with hyperlinks
if all_wine_data:
    # --- Convert price/stock fields to numeric types and sort ascending
    def _to_float_or_none(x):
        try:
            return float(x)
        except Exception:
            return None

    def _to_int_or_none(x):
        try:
            return int(float(x))
        except Exception:
            return None

    for it in all_wine_data:
        # Convert main numeric fields in-place so DataFrame receives numeric dtypes
        it['sellPrice'] = _to_float_or_none(it.get('sellPrice'))
        it['productOriginalPrice'] = _to_float_or_none(it.get('productOriginalPrice'))
        it['stock'] = _to_int_or_none(it.get('stock'))

        # Prepare sort keys: missing values should sort AFTER valid entries when sorting ascending
        dr = it.get('discount_rate')
        try:
            it['_discount_for_sort'] = float(dr) if dr is not None else float('inf')
        except Exception:
            it['_discount_for_sort'] = float('inf')

        it['_sell_for_sort'] = it['sellPrice'] if it.get('sellPrice') is not None else float('inf')

    # Sort in-place: lowest discount_rate first, then lowest sellPrice
    all_wine_data.sort(key=lambda x: (x.get('_discount_for_sort', float('inf')), x.get('_sell_for_sort', float('inf'))))

    # Prepare main data (without temp fields)
    main_data = []
    local_paths = []
    web_urls = []
    
    for item in all_wine_data:
        main_data.append({
            k: v for k, v in item.items() 
            if not k.startswith('_')
        })
        local_paths.append(item.get('_local_image_path', ''))
        web_urls.append(item.get('_web_image_url', ''))
    
    df = pd.DataFrame(main_data, columns=excel_columns)
    excel_filename = "wine_data.xlsx"
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Wine Data')
        worksheet = writer.sheets['Wine Data']
        
        # Apply number formats:
        # - sellPrice (col 4) and productOriginalPrice (col 5): two decimals
        # - discount_rate (col 6): percent with two decimals
        # - stock (col 7): integer
        for idx in range(len(df)):
            # sellPrice (col 4)
            sp = df.iloc[idx].get('sellPrice')
            if pd.notnull(sp):
                cell_sp = worksheet.cell(row=idx+2, column=4)
                try:
                    cell_sp.value = float(sp)
                except Exception:
                    pass
                cell_sp.number_format = '0.00'

            # productOriginalPrice (col 5)
            pop = df.iloc[idx].get('productOriginalPrice')
            if pd.notnull(pop):
                cell_pop = worksheet.cell(row=idx+2, column=5)
                try:
                    cell_pop.value = float(pop)
                except Exception:
                    pass
                cell_pop.number_format = '0.00'

            # discount_rate (col 6)
            val = df.iloc[idx]["discount_rate"]
            if pd.notnull(val):
                cell = worksheet.cell(row=idx+2, column=6)
                try:
                    cell.value = float(val)
                except Exception:
                    pass
                cell.number_format = '0.00%'

            # stock (col 7)
            st = df.iloc[idx].get('stock')
            if pd.notnull(st):
                cell_st = worksheet.cell(row=idx+2, column=7)
                try:
                    # ensure integer presentation when possible
                    cell_st.value = int(st)
                except Exception:
                    try:
                        cell_st.value = float(st)
                    except Exception:
                        pass
                cell_st.number_format = '0'

        # Add hyperlinks to image_name column (col 11 = K)
        for idx, local_path in enumerate(local_paths):
            if local_path:
                cell = worksheet.cell(row=idx+2, column=11)
                cell.value = df.iloc[idx]["image_name"]
                # Use file URI for local paths so Excel can open them reliably
                try:
                    uri = Path(local_path).resolve().as_uri()
                except Exception:
                    uri = local_path
                cell.hyperlink = uri
                cell.style = "Hyperlink"

        # Add hyperlinks to image_url column (col 12 = L)
        for idx, web_url in enumerate(web_urls):
            if web_url:
                cell = worksheet.cell(row=idx+2, column=12)
                cell.value = df.iloc[idx]["image_url"]
                cell.hyperlink = web_url
                cell.style = "Hyperlink"
    
    print(f"\n✅ Completed! Processed {len(all_wine_data)} wine items.")
    print(f"📁 Data saved to: {os.path.abspath(excel_filename)}")
    print(f"🖼️  Images saved to: {images_dir.resolve()}")
else:
    print("\n⚠️  No data was processed.")
