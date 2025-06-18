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
    # Extracts and renames columns based on the selected header row
    # Reverted: Load entire file from top
    try:
        df = pd.read_excel(file)
    except:
        df = pd.read_csv(file)

    # Clean whitespace from all cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    

    mapping = {
        'Type': df.columns[0] if len(df.columns) > 0 else None,
        'Count': df.columns[1] if len(df.columns) > 1 else None,
        'Weight': df.columns[2] if len(df.columns) > 2 else None,
        'Height': df.columns[3] if len(df.columns) > 3 else None,
        'Width': df.columns[4] if len(df.columns) > 4 else None,
        'Depth': df.columns[5] if len(df.columns) > 5 else None
    }
    st.subheader("Adjust Column Mapping (optional)")
    use_defaults = st.checkbox("Use default column mapping (ignore smart detection)", value=False)

    if use_defaults:
        mapping = {
            'Type': df.columns[0] if len(df.columns) > 0 else None,
            'Count': df.columns[1] if len(df.columns) > 1 else None,
            'Weight': df.columns[2] if len(df.columns) > 2 else None,
            'Height': df.columns[3] if len(df.columns) > 3 else None,
            'Width': df.columns[4] if len(df.columns) > 4 else None,
            'Depth': df.columns[5] if len(df.columns) > 5 else None
        }
    type_col = st.selectbox("Column for Type", df.columns, index=df.columns.get_loc(mapping['Type']) if mapping.get('Type') in df.columns else 0)
    count_col = st.selectbox("Column for Count", df.columns, index=df.columns.get_loc(mapping['Count']) if mapping.get('Count') in df.columns else 0)
    height_col = st.selectbox("Column for Height", df.columns, index=df.columns.get_loc(mapping['Height']) if mapping.get('Height') in df.columns else 0)
    width_col = st.selectbox("Column for Width", df.columns, index=df.columns.get_loc(mapping['Width']) if mapping.get('Width') in df.columns else 0)
    depth_col = st.selectbox("Column for Depth", df.columns, index=df.columns.get_loc(mapping['Depth']) if mapping.get('Depth') in df.columns else 0)
    weight_col = st.selectbox(
        "Column for Weight (optional)",
        ["None"] + df.columns.tolist(),
        index=(df.columns.get_loc(mapping['Weight']) + 1) if mapping.get('Weight') and mapping['Weight'] in df.columns else 0
    )

    selected_cols = [type_col, count_col, height_col, width_col, depth_col]
    new_names = ['Type', 'Count', 'Height', 'Width', 'Depth']
    if weight_col != "None":
        selected_cols.append(weight_col)
        new_names.append('Weight')

    try:
        extracted = df[selected_cols]
        extracted.columns = new_names
        extracted = extracted.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        extracted.replace("", pd.NA, inplace=True)
        extracted = extracted.dropna(how='any')

        header_keywords = ['type', 'tips', 'count', 'qty', 'skaits', 'height', 'augstums', 'width', 'platums', 'garums', 'depth', 'dziļums', 'weight', 'svars']
        extracted = extracted[~extracted.apply(lambda row: sum(any(str(val).lower() == kw for kw in header_keywords) for val in row) >= 3, axis=1)]

        return extracted
    except Exception as e:
        st.error(f"Failed to extract data using selected columns. Error: {e}")
        return pd.DataFrame()

# ---- HANDLE UPLOADED FILE ----

def compute_beds_and_trucks(panels, bed_width=2400, bed_weight_limit=2500, truck_weight_limit=15000, truck_max_length=13620):
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
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        df = extract_from_pdf(uploaded_file)
    else:
        df = extract_from_excel_or_csv(uploaded_file)

    if not df.empty:
        st.success("Data extracted successfully!")

        st.subheader("Row Removal and Update Option")
        delete_rows = st.multiselect("Select row indices to delete from extracted data", df.index.tolist(), default=[df.index[0]] if not df.empty else [])
        amend_data = st.checkbox("Amend extracted data after row deletion", value=True)
        if amend_data and delete_rows:
            df = df.drop(delete_rows).reset_index(drop=True)
            st.info("Selected rows have been removed and data updated.")

        st.subheader("Extracted GRC Panel Data")
        st.dataframe(df)

        if 'Count' in df.columns:
            total = df['Count'].sum()
            st.markdown(f"**Total Panel Count:** {total}")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Extracted Data as CSV",
            data=csv,
            file_name='extracted_grc_panels.csv',
            mime='text/csv'
        )

        # ---- BED & TRUCK PACKING ----
        # Display configuration inputs
        bed_width = st.number_input("Bed Width (mm)", value=2400)
        bed_weight_limit = st.number_input("Bed Weight Limit (kg)", value=2500)
        truck_weight_limit = st.number_input("Truck Weight Limit (kg)", value=15000)
        truck_max_length = st.number_input("Truck Max Length (mm)", value=13620)
        panel_thickness = st.number_input("Panel Thickness (mm, if no weight provided)", value=30)
        density = 2100  # kg/m3

        
        with st.spinner("Running transport analysis..."):
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

            beds, trucks = compute_beds_and_trucks(panel_rows, bed_width, bed_weight_limit, truck_weight_limit, truck_max_length)

            # Validation warnings
            overfilled_beds = [i for i, bed in enumerate(beds) if bed['Width'] > bed_width or bed['Weight'] > bed_weight_limit]
            if overfilled_beds:
                st.warning(f"⚠️ Overfilled Beds: {overfilled_beds}")

            overfilled_trucks = [i for i, truck in enumerate(trucks) if sum(b['Length'] for b in truck) > truck_max_length or sum(b['Weight'] for b in truck) > truck_weight_limit]
            if overfilled_trucks:
                st.warning(f"⚠️ Overfilled Trucks: {overfilled_trucks}")

            st.warning(f"⚠️ Overfilled Trucks: {overfilled_trucks}")
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

            beds, trucks = compute_beds_and_trucks(panel_rows, bed_width, bed_weight_limit, truck_weight_limit, truck_max_length)
            st.markdown(f"**Total Beds:** {len(beds)}")
            st.markdown(f"**Total Trucks:** {len(trucks)}")

            # Bed usage visualization
            import matplotlib.pyplot as plt

            usage_data = pd.DataFrame(beds)
            usage_data['Used Width (mm)'] = usage_data['Width']
            usage_data['Used Weight (kg)'] = usage_data['Weight']

            fig1, ax1 = plt.subplots()
            ax1.bar(range(len(beds)), usage_data['Used Width (mm)'])
            ax1.set_title("Used Bed Width per Bed")
            ax1.set_xlabel("Bed Index")
            ax1.set_ylabel("Width (mm)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.bar(range(len(beds)), usage_data['Used Weight (kg)'])
            ax2.set_title("Used Bed Weight per Bed")
            ax2.set_xlabel("Bed Index")
            ax2.set_ylabel("Weight (kg)")
            st.pyplot(fig2)

            # Truck usage visualization
            truck_weights = [sum(b['Weight'] for b in truck) for truck in trucks]
            truck_lengths = [sum(b['Length'] for b in truck) for truck in trucks]

            fig3, ax3 = plt.subplots()
            ax3.bar(range(len(trucks)), truck_weights)
            ax3.set_title("Total Weight per Truck")
            ax3.set_xlabel("Truck Index")
            ax3.set_ylabel("Weight (kg)")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots()
            ax4.bar(range(len(trucks)), truck_lengths)
            ax4.set_title("Used Length per Truck")
            ax4.set_xlabel("Truck Index")
            ax4.set_ylabel("Length (mm)")
            st.pyplot(fig4)

            for i, truck in enumerate(trucks):
                st.markdown(f"### Truck {i+1} - Beds: {len(truck)}")
                truck_df = pd.DataFrame(truck)
                st.dataframe(truck_df)

            export_data = []
            for i, truck in enumerate(trucks):
                for j, bed in enumerate(truck):
                    export_data.append({
                        'Truck': i + 1,
                        'Bed': j + 1,
                        'Length': bed['Length'],
                        'Height': bed['Height'],
                        'Width': bed['Width'],
                        'Weight': bed['Weight'],
                        'Num Panels': bed['Num Panels'],
                        'Panel Types': ', '.join(bed['Panel Types'])
                    })

            export_df = pd.DataFrame(export_data)
            st.download_button(
                label="Download Packing Plan as CSV",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name='truck_packing_plan.csv',
                mime='text/csv'
            )

            excel_buffer = pd.ExcelWriter("packing_plan.xlsx", engine='xlsxwriter')
            export_df.to_excel(excel_buffer, index=False, sheet_name='PackingPlan')

            # Add separate truck sheets
            for i, truck in enumerate(trucks):
                pd.DataFrame(truck).to_excel(excel_buffer, index=False, sheet_name=f'Truck {i+1}')

            # Create summary of unique beds
            bed_summary_df = pd.DataFrame(beds)
            unique_beds = bed_summary_df.groupby(['Length', 'Height', 'Width']).size().reset_index(name='Quantity')
            unique_beds.to_excel(excel_buffer, index=False, sheet_name='Bed Summary')

            excel_buffer.close()

            with open("packing_plan.xlsx", "rb") as f:
                st.download_button(
                    label="Download Packing Plan as Excel",
                    data=f.read(),
                    file_name="truck_packing_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            st.success("Packing plan generated with separate sheets for trucks and a bed summary.")

            st.download_button(
                    label="Download Packing Plan as Excel",
                    data=f.read(),
                    file_name="truck_packing_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error computing transport plan: {e}")
