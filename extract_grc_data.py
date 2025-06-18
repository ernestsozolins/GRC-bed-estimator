import streamlit as st
import pandas as pd
import pdfplumber
import re

st.title("GRC Panel Specification Extractor")

uploaded_file = st.file_uploader("Upload a PDF, Excel, or CSV file", type=["pdf", "xlsx", "xls", "csv"])

def extract_from_pdf(file):
    data = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            matches = re.findall(r'(Grc\.[\w\.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', text)
            for match in matches:
                unit_type, count, height, width, depth = match
                data.append({
                    "Type": unit_type,
                    "Count": int(count),
                    "Height": int(height),
                    "Width": int(width),
                    "Depth": int(depth),
                    "Weight": None  # Optional column
                })
    return pd.DataFrame(data)


def extract_from_excel_or_csv(file):
    try:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.replace("", pd.NA, inplace=True)
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Failed to extract table: {e}")
        return pd.DataFrame()

# ---- MAIN APP LOGIC ----
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        df = extract_from_pdf(uploaded_file)
    else:
        df = extract_from_excel_or_csv(uploaded_file)

    if not df.empty:
        st.success("Data extracted successfully!")
        st.dataframe(df)

        # Transport Configuration
        st.subheader("Transport Configuration")
        bed_width = st.number_input("Bed Width (mm)", value=2400)
        bed_weight_limit = st.number_input("Bed Weight Limit (kg)", value=2500)
        truck_weight_limit = st.number_input("Truck Weight Limit (kg)", value=15000)
        truck_max_length = st.number_input("Truck Max Length (mm)", value=13620)
        panel_thickness = st.number_input("Panel Thickness (mm, if no weight provided)", value=30)
        density = 2100  # kg/m3

        run_analysis = st.button("Run Transport Analysis")
        if run_analysis:
            st.subheader("Transport Packing Plan")
            try:
                panel_rows = []
                for _, row in df.iterrows():
                    count = int(row['Count']) if 'Count' in row and pd.notna(row['Count']) else 1
                    for _ in range(count):
                        panel_rows.append({
                            'Type': row['Type'],
                            'Height': float(row['Height']),
                            'Width': float(row['Width']),
                            'Depth': float(row['Depth']),
                            'Weight': float(row['Weight']) if 'Weight' in row and pd.notna(row['Weight']) else float(row['Height']) * float(row['Width']) * (panel_thickness / 1000) * (density / 1000)
                        })

                def compute_beds_and_trucks(panels):
                    beds = []
                    for panel in panels:
                        placed = False
                        for bed in beds:
                            used_depth = sum(p['Depth'] for p in bed)
                            total_weight = sum(p['Weight'] for p in bed)
                            if used_depth + panel['Depth'] <= bed_width and total_weight + panel['Weight'] <= bed_weight_limit:
                                bed.append(panel)
                                placed = True
                                break
                        if not placed:
                            beds.append([panel])

                    bed_summaries = []
                    for bed in beds:
                        bed_length = max(p['Width'] for p in bed)
                        bed_height = max(p['Height'] for p in bed)
                        bed_weight = sum(p['Weight'] for p in bed)
                        panel_types = [str(p['Type']).strip() for p in bed if pd.notna(p['Type']) and isinstance(p['Type'], str) and str(p['Type']).strip().lower() not in ("", "nan", "none")]
                        bed_summaries.append({
                            'Length': bed_length,
                            'Height': bed_height,
                            'Width': bed_width,
                            'Weight': bed_weight,
                            'Num Panels': len(bed),
                            'Panel Types': panel_types
                        })

                    trucks = []
                    for bed in bed_summaries:
                        placed = False
                        for truck in trucks:
                            used_length = sum(b['Length'] for b in truck)
                            total_weight = sum(b['Weight'] for b in truck)
                            if used_length + bed['Length'] <= truck_max_length and total_weight + bed['Weight'] <= truck_weight_limit:
                                truck.append(bed)
                                placed = True
                                break
                        if not placed:
                            trucks.append([bed])

                    return bed_summaries, trucks

                beds, trucks = compute_beds_and_trucks(panel_rows)
                st.markdown(f"**Total Beds:** {len(beds)}")
                st.markdown(f"**Total Trucks:** {len(trucks)}")

                for i, truck in enumerate(trucks):
                st.markdown(f"### Truck {i+1} - Beds: {len(truck)}")
                truck_df = pd.DataFrame(truck)
                st.dataframe(truck_df)

                # Truck utilization visualization
                import matplotlib.pyplot as plt
                used_length = truck_df['Length'].sum()
                used_weight = truck_df['Weight'].sum()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                ax1.barh(["Truck Length"], [used_length], color='skyblue')
                ax1.set_xlim(0, truck_max_length)
                ax1.set_title("Truck Length Used (mm)")
                ax2.barh(["Truck Weight"], [used_weight], color='lightcoral')
                ax2.set_xlim(0, truck_weight_limit)
                ax2.set_title("Truck Weight Used (kg)")
                st.pyplot(fig)

            # Summary table for unique beds
            bed_df = pd.DataFrame(beds)
            bed_summary = bed_df.groupby(['Length', 'Height', 'Width']).size().reset_index(name='Quantity')
            st.subheader("Unique Bed Summary")
            st.dataframe(bed_summary)

            # Export to Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                bed_df.to_excel(writer, index=False, sheet_name='Beds')
                bed_summary.to_excel(writer, index=False, sheet_name='Bed Summary')
                for i, truck in enumerate(trucks):
                    pd.DataFrame(truck).to_excel(writer, index=False, sheet_name=f'Truck {i+1}')
            st.download_button(
                label="Download Packing Plan (Excel)",
                data=output.getvalue(),
                file_name="transport_packing_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            except Exception as e:
                st.error(f"Error computing transport plan: {e}")
