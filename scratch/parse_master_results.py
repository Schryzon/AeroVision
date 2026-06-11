# scratch/parse_master_results.py
import json

with open("AeroVision.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== PARSING MASTER COMPARING RESULTS ===")
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        # Find cell that prints the final results dataframe
        if "pd.DataFrame" in source and "accuracy" in source or "print(df_results)" in source or "df_results" in source:
            for out in outputs:
                if out.get("output_type") == "stream" and out.get("name") == "stdout":
                    print("".join(out.get("text", [])))
                elif out.get("output_type") == "execute_result":
                    data = out.get("data", {})
                    if "text/plain" in data:
                        print("".join(data["text/plain"]))
